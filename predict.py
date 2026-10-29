"""Load saved weights and classify new Raman cell maps."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import iter_maps, make_bag, normalize, write_csv
from src.models import build_model, probabilities


def load_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("classes") != {"ALL": 0, "AML": 1}:
        raise ValueError("Unsupported checkpoint format or class mapping.")
    model = build_model(checkpoint["model_name"], checkpoint["input_channels"], checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def predict_maps(checkpoint_path, source_path, output_path, device="cpu"):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    channel_mean = checkpoint["normalizer_mean"].numpy()
    channel_std = checkpoint["normalizer_std"].numpy()
    prediction_rows = []
    for file_name, table in iter_maps(source_path):
        patch_bag = make_bag(table, checkpoint["config"]["preprocessing"])
        normalized_bags = normalize(patch_bag[None, :, :], channel_mean, channel_std)
        with torch.no_grad():
            aml_probabilities, predicted_labels = probabilities(
                model, checkpoint["model_name"], torch.as_tensor(normalized_bags, device=device)
            )
        aml_probability = float(aml_probabilities.cpu().item())
        if not np.isfinite(aml_probability):
            raise ValueError(f"Nonfinite prediction: {file_name}")
        prediction_rows.append({
            "file_name": file_name,
            "predicted_label": "AML" if predicted_labels.item() else "ALL",
            "probability_aml": aml_probability,
        })
    write_csv(output_path, prediction_rows)
    return prediction_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="A Feather file, tar archive, or folder of maps.",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    prediction_rows = predict_maps(args.checkpoint, args.input, args.output)
    print(f"Saved {len(prediction_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
