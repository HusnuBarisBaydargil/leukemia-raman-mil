"""Train selected MIL models and automatically save one checkpoint per outer fold."""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn

from src.data import (
    ROOT,
    inner_partition,
    load_bags,
    load_config,
    normalize,
    read_csv,
    write_csv,
)
from src.models import build_model, probabilities


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_splits(split_rows):
    if {row["fold_id"] for row in split_rows} != {"1", "2", "3", "4"}:
        raise ValueError("Expected the fixed four-fold split.")
    fold_file_keys = [(row["fold_id"], row["file_name"]) for row in split_rows]
    if len(set(fold_file_keys)) != len(split_rows):
        raise ValueError("Duplicate file within an outer fold.")
    file_names = {row["file_name"] for row in split_rows}
    file_identities = {}
    for row in split_rows:
        if row["split_role"] not in {"train", "test"} or row["aml_all_label"] not in {"ALL", "AML"}:
            raise ValueError("Unexpected split role or label.")
        file_identity = row["patient_id"], row["aml_all_label"]
        if file_identities.setdefault(row["file_name"], file_identity) != file_identity:
            raise ValueError("File identity differs across folds.")
    heldout_rows = [row for row in split_rows if row["split_role"] == "test"]
    if Counter(row["file_name"] for row in heldout_rows) != Counter(file_names):
        raise ValueError("Each map must be held out exactly once.")
    for fold_id in range(1, 5):
        fold_rows = [row for row in split_rows if row["fold_id"] == str(fold_id)]
        train_rows = [row for row in fold_rows if row["split_role"] == "train"]
        test_rows = [row for row in fold_rows if row["split_role"] == "test"]
        if {row["file_name"] for row in fold_rows} != file_names:
            raise ValueError("Incomplete outer fold.")
        if {row["patient_id"] for row in train_rows} & {row["patient_id"] for row in test_rows}:
            raise ValueError("Train/test group overlap.")
        if any(
            {row["aml_all_label"] for row in partition} != {"ALL", "AML"}
            for partition in [train_rows, test_rows]
        ):
            raise ValueError("Each partition must contain both classes.")


def fit(
    model_name,
    train_bags,
    train_labels,
    validation_bags,
    validation_labels,
    config,
    fold,
    device,
    soft_targets=None,
    stage_offset=0,
):
    training_settings = {**config["training"], **config["model_overrides"].get(model_name, {})}
    set_seed(config["seed"] + fold + training_settings.get("seed_offset", 0) + stage_offset)
    model = build_model(model_name, train_bags.shape[-1], config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_settings["learning_rate"],
        weight_decay=training_settings["weight_decay"],
    )
    train_inputs = torch.as_tensor(train_bags, dtype=torch.float32, device=device)
    validation_inputs = torch.as_tensor(validation_bags, dtype=torch.float32, device=device)
    train_label_tensor = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    validation_label_tensor = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    class_counts = np.bincount(train_labels, minlength=2)
    if (class_counts == 0).any():
        raise ValueError("Inner training needs both ALL and AML.")
    positive_class_weight = torch.tensor(
        [class_counts[0] / class_counts[1]], dtype=torch.float32, device=device
    )
    class_weights = torch.tensor(len(train_labels) / (2 * class_counts), dtype=torch.float32, device=device)
    training_targets = (
        train_label_tensor.float()
        if soft_targets is None
        else torch.as_tensor(soft_targets, dtype=torch.float32, device=device)
    )

    def loss(bags, hard_labels, target=None, auxiliary=False):
        if model_name in {"abmil", "clam-sb"}:
            model_outputs = model(bags, hard_labels) if model_name == "clam-sb" else model(bags)
            bag_loss = nn.functional.cross_entropy(
                model_outputs["bag_logits"], hard_labels, weight=class_weights
            )
            if model_name == "clam-sb" and auxiliary:
                return 0.7 * bag_loss + 0.3 * model_outputs["instance_loss"]
            return bag_loss
        bag_logits = model(bags)[0]
        return nn.functional.binary_cross_entropy_with_logits(
            bag_logits, hard_labels.float() if target is None else target, pos_weight=positive_class_weight
        )

    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    training_history = []
    for epoch in range(1, training_settings["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss(train_inputs, train_label_tensor, training_targets, auxiliary=True)
        if not torch.isfinite(train_loss):
            raise ValueError("Nonfinite training loss.")
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training_settings["grad_clip_norm"])
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss(validation_inputs, validation_label_tensor).item())
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss.")
        training_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation_loss - training_settings["min_delta"]:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = {
                parameter_name: parameter_tensor.detach().cpu().clone()
                for parameter_name, parameter_tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epoch >= training_settings["min_epochs"]
                and epochs_without_improvement >= training_settings["patience"]
            ):
                break
    if best_model_state is None:
        raise ValueError("No valid model checkpoint was selected.")
    model.load_state_dict(best_model_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "history": training_history,
    }


def partition_data(partition_rows, bag_lookup, channel_mean, channel_std):
    normalized_bags = normalize(
        np.stack([bag_lookup[row["file_name"]] for row in partition_rows]),
        channel_mean,
        channel_std,
    )
    class_labels = np.asarray([int(row["aml_all_label"] == "AML") for row in partition_rows], dtype=np.int64)
    return normalized_bags, class_labels


def prepare_fold(split_rows, bag_lookup, fold_id, seed):
    """Partition one outer fold and fit normalization on its inner training rows."""
    outer_train_rows = [
        row for row in split_rows
        if row["fold_id"] == str(fold_id) and row["split_role"] == "train"
    ]
    test_rows = [
        row for row in split_rows
        if row["fold_id"] == str(fold_id) and row["split_role"] == "test"
    ]
    inner_train_rows, validation_rows = inner_partition(outer_train_rows, fold_id, seed)
    train_instances = np.concatenate([bag_lookup[row["file_name"]] for row in inner_train_rows], axis=0)
    channel_mean = train_instances.mean(axis=0, dtype=np.float64).astype(np.float32)
    channel_std = train_instances.std(axis=0, dtype=np.float64).astype(np.float32)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std).astype(np.float32)
    return {
        "outer_train_rows": outer_train_rows,
        "inner_rows": inner_train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "mean": channel_mean,
        "std": channel_std,
    }


def make_soft_targets(probabilities_aml, hard_labels, outer_train_rows, inner_train_rows, smoothing):
    """Build entropy-smoothed targets in inner-training row order."""
    clipped_probabilities = np.clip(
        probabilities_aml.astype(np.float64),
        smoothing["entropy_epsilon"],
        1 - smoothing["entropy_epsilon"],
    )
    predictive_entropy = (
        -clipped_probabilities * np.log(clipped_probabilities)
        - (1 - clipped_probabilities) * np.log(1 - clipped_probabilities)
    )
    smoothing_strength = np.clip(
        smoothing["alpha_max"] * predictive_entropy / np.log(2),
        0,
        smoothing["alpha_max"],
    )
    smoothed_targets = (1 - smoothing_strength) * hard_labels + smoothing_strength * 0.5
    targets_by_file = {
        row["file_name"]: soft_target
        for row, soft_target in zip(outer_train_rows, smoothed_targets)
    }
    return np.asarray([targets_by_file[row["file_name"]] for row in inner_train_rows], dtype=np.float32)


def save_fold_checkpoint(model, model_name, fold_id, config, fold_data, training_details, model_output_dir):
    checkpoint = {
        "format_version": 1,
        "model_name": model_name,
        "fold": fold_id,
        "config": config,
        "input_channels": fold_data["mean"].shape[0],
        "classes": {"ALL": 0, "AML": 1},
        "state_dict": {
            parameter_name: parameter_tensor.detach().cpu()
            for parameter_name, parameter_tensor in model.state_dict().items()
        },
        "normalizer_mean": torch.from_numpy(fold_data["mean"]),
        "normalizer_std": torch.from_numpy(fold_data["std"]),
        "inner_train_groups": sorted({row["patient_id"] for row in fold_data["inner_rows"]}),
        "validation_groups": sorted({row["patient_id"] for row in fold_data["validation_rows"]}),
        "test_groups": sorted({row["patient_id"] for row in fold_data["test_rows"]}),
        "best_epoch": training_details["best_epoch"],
    }
    checkpoint_path = model_output_dir / f"fold_{fold_id}.pt"
    torch.save(checkpoint, checkpoint_path)
    history_path = model_output_dir / f"fold_{fold_id}_training.json"
    history_path.write_text(json.dumps(training_details, indent=2), encoding="utf-8")
    return checkpoint_path


def evaluate_fold(model, model_name, fold_id, fold_data, bag_lookup, device):
    test_rows = fold_data["test_rows"]
    test_inputs, test_labels = partition_data(test_rows, bag_lookup, fold_data["mean"], fold_data["std"])
    with torch.no_grad():
        aml_probabilities, predicted_labels = probabilities(
            model, model_name, torch.as_tensor(test_inputs, device=device)
        )
    aml_probabilities, predicted_labels = aml_probabilities.cpu().numpy(), predicted_labels.cpu().numpy()
    prediction_rows = []
    for row, predicted_label, aml_probability, true_label in zip(
        test_rows, predicted_labels, aml_probabilities, test_labels
    ):
        prediction_rows.append({
            "file_name": row["file_name"],
            "patient_id": row["patient_id"],
            "fold": fold_id,
            "true_label": "AML" if true_label else "ALL",
            "predicted_label": "AML" if predicted_label else "ALL",
            "probability_aml": float(aml_probability),
        })
    fold_metrics = {
        "fold": fold_id,
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted_labels)),
        "macro_f1": float(f1_score(test_labels, predicted_labels, average="macro")),
    }
    return prediction_rows, fold_metrics


def train_fold(model_name, split_rows, bag_lookup, config, fold_id, device, model_output_dir):
    fold_data = prepare_fold(split_rows, bag_lookup, fold_id, config["seed"])
    channel_mean, channel_std = fold_data["mean"], fold_data["std"]
    train_inputs, train_labels = partition_data(
        fold_data["inner_rows"], bag_lookup, channel_mean, channel_std
    )
    validation_inputs, validation_labels = partition_data(
        fold_data["validation_rows"], bag_lookup, channel_mean, channel_std
    )
    model, training_details = fit(
        model_name, train_inputs, train_labels, validation_inputs, validation_labels,
        config, fold_id, device,
    )
    if model_name == "ug-dec-mil":
        outer_train_inputs, outer_train_labels = partition_data(
            fold_data["outer_train_rows"], bag_lookup, channel_mean, channel_std
        )
        with torch.no_grad():
            stage_a_probabilities, _ = probabilities(
                model, model_name, torch.as_tensor(outer_train_inputs, device=device)
            )
        soft_targets = make_soft_targets(
            stage_a_probabilities.cpu().numpy(),
            outer_train_labels,
            fold_data["outer_train_rows"],
            fold_data["inner_rows"],
            config["smoothing"],
        )
        stage_a_best_epoch = training_details["best_epoch"]
        model, training_details = fit(
            model_name, train_inputs, train_labels, validation_inputs, validation_labels,
            config, fold_id, device,
            soft_targets=soft_targets,
            stage_offset=config["smoothing"]["stage_b_seed_offset"],
        )
        training_details["stage_a_best_epoch"] = stage_a_best_epoch
    checkpoint_path = save_fold_checkpoint(
        model, model_name, fold_id, config, fold_data, training_details, model_output_dir
    )
    prediction_rows, fold_metrics = evaluate_fold(model, model_name, fold_id, fold_data, bag_lookup, device)
    print(
        f"{model_name}: fold {fold_id}, "
        f"balanced accuracy {fold_metrics['balanced_accuracy']:.4f}; saved {checkpoint_path}",
        flush=True,
    )
    return prediction_rows, fold_metrics


def summarize_model(model_name, prediction_rows, fold_metrics):
    true_labels = [int(row["true_label"] == "AML") for row in prediction_rows]
    predicted_labels = [int(row["predicted_label"] == "AML") for row in prediction_rows]
    return {
        "model": model_name,
        "fold_mean_balanced_accuracy": float(np.mean([
            row["balanced_accuracy"] for row in fold_metrics
        ])),
        "fold_mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_metrics])),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "pooled_macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
    }


def train_models(config, model_names, cache_dir, split_path, output_dir):
    split_rows = read_csv(split_path)
    validate_splits(split_rows)
    bag_lookup = load_bags(cache_dir, split_rows, config["preprocessing"])
    output_dir = Path(output_dir)
    for model_name in model_names:
        if model_name not in config["models"]:
            raise ValueError(f"Unknown model: {model_name}")
        if (output_dir / model_name).exists():
            raise FileExistsError(
                f"Model folder already exists: {(output_dir / model_name).resolve()}. "
                "It may contain a completed or partial run. "
                "Existing weights were left unchanged. To restart this model, rename its folder as a backup "
                f"or set paths.outputs in config.json to a new folder, then rerun with --model {model_name}."
            )
    model_summaries = []
    for model_name in model_names:
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True)
        prediction_rows, fold_metrics = [], []
        current_stage = "selecting the device"
        try:
            requested_device = config["model_overrides"].get(model_name, {}).get("device", config["device"])
            device = torch.device(
                ("cuda" if torch.cuda.is_available() else "cpu")
                if requested_device == "auto" else requested_device
            )
            for fold_id in range(1, 5):
                current_stage = f"fold {fold_id}"
                fold_predictions, fold_result = train_fold(
                    model_name, split_rows, bag_lookup, config, fold_id, device, model_output_dir
                )
                prediction_rows.extend(fold_predictions)
                fold_metrics.append(fold_result)
            current_stage = "writing final results"
            write_csv(model_output_dir / "predictions.csv", prediction_rows)
            write_csv(model_output_dir / "fold_metrics.csv", fold_metrics)
            model_summaries.append(summarize_model(model_name, prediction_rows, fold_metrics))
            write_csv(model_output_dir / "metrics.csv", [model_summaries[-1]])
        except (Exception, KeyboardInterrupt) as error:
            error.add_note(
                f"Training for {model_name} did not finish during {current_stage}. "
                f"Partial outputs were preserved in {model_output_dir.resolve()}. "
                "Already-written checkpoints remain available, but the interrupted file may be incomplete. "
                "To restart this model from the beginning, rename its folder as a backup "
                "or set paths.outputs "
                f"in config.json to a new folder, then rerun with --model {model_name}. "
                "Other model folders were left unchanged; runs do not resume automatically."
            )
            raise
    return model_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="all",
        help="ug-dec-mil, dec-mil, attention-mil, abmil, clam-sb, or all (default).",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON configuration; defaults to config.json.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        selected_models = config["models"] if args.model == "all" else [args.model]
        train_models(
            config,
            selected_models,
            ROOT / config["paths"]["cache"],
            ROOT / "data/splits.csv",
            ROOT / config["paths"]["outputs"],
        )
    except (Exception, KeyboardInterrupt) as error:
        was_interrupted = isinstance(error, KeyboardInterrupt)
        print("Training interrupted." if was_interrupted else f"Training failed: {error}", file=sys.stderr)
        for note in getattr(error, "__notes__", []):
            print(note, file=sys.stderr)
        return 130 if was_interrupted else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
