"""Synthetic preprocessing -> five-model training -> weight reload -> inference."""

import copy
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import load_checkpoint, predict_maps
from src.data import inner_partition, load_config, make_bag, normalize, prepare_data, read_csv, write_csv
from src.models import probabilities
from train import train_models, validate_splits


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generated fixtures stay under ignored outputs; no source files are changed.
        generated = ROOT / "outputs"
        generated.mkdir(exist_ok=True)
        cls.fixture = Path(tempfile.mkdtemp(prefix="test_workflow_", dir=generated))
        cls.raw = cls.fixture / "raw"
        cls.raw.mkdir()
        cls.config = load_config()
        cls.config["preprocessing"] = {"map_shape": [8, 8], "channels": 16, "patch_size": 1}
        cls.config["device"] = "cpu"
        cls.config["training"].update(max_epochs=2, min_epochs=1, patience=1)
        manifest, split = [], []
        x, y = np.meshgrid(np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32))
        for label_index, label in enumerate(["ALL", "AML"]):
            for group in range(1, 9):
                name = f"{label}_{group}.feather"
                manifest.append({"file_name": name, "patient_id": f"{label}__{group}",
                                 "aml_all_label": label, "artifact_status": "unknown"})
                columns = {str(500 + channel): pa.array((x + y * 8 + channel + group + label_index * 0.2).astype(np.float32).ravel())
                           for channel in range(16)}
                columns.update(x_pos=pa.array(x.ravel()), y_pos=pa.array(y.ravel()))
                feather.write_feather(pa.table(columns), cls.raw / name)
                for fold in range(1, 5):
                    split.append({**manifest[-1], "fold_id": str(fold),
                                  "split_role": "test" if (group - 1) % 4 == fold - 1 else "train"})
        cls.split = sorted(split, key=lambda r: (r["fold_id"], r["split_role"], r["file_name"]))
        cls.split_path = cls.fixture / "splits.csv"
        cls.manifest_path = cls.fixture / "manifest.csv"
        write_csv(cls.split_path, cls.split)
        write_csv(cls.manifest_path, manifest)
        cls.cache = cls.fixture / "cache"
        prepare_data(cls.raw, cls.cache, cls.manifest_path, cls.config["preprocessing"])
        cls.output = cls.fixture / "outputs"
        cls.metrics = train_models(cls.config, cls.config["models"], cls.cache, cls.split_path, cls.output)

    def test_all_five_models_save_weights_and_predict(self):
        self.assertEqual(len(self.metrics), 5)
        sample = self.raw / "ALL_1.feather"
        bag = make_bag(feather.read_table(sample), self.config["preprocessing"])
        for name in self.config["models"]:
            with self.subTest(model=name):
                self.assertEqual(len(list((self.output / name).glob("fold_*.pt"))), 4)
                model, checkpoint = load_checkpoint(self.output / name / "fold_1.pt")
                self.assertFalse(set(checkpoint["inner_train_groups"]) & set(checkpoint["test_groups"]))
                prepared = normalize(bag[None], checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_std"].numpy())
                with torch.no_grad():
                    expected, label = probabilities(model, name, torch.from_numpy(prepared))
                result = predict_maps(self.output / name / "fold_1.pt", sample, self.fixture / f"{name}_predictions.csv")
                self.assertAlmostEqual(result[0]["probability_aml"], expected.item(), places=7)
                self.assertEqual(result[0]["predicted_label"], "AML" if label.item() else "ALL")

    def test_heldout_checkpoint_matches_training_prediction(self):
        sample = self.raw / "ALL_1.feather"
        for name in self.config["models"]:
            with self.subTest(model=name):
                expected = next(r for r in read_csv(self.output / name / "predictions.csv") if r["file_name"] == sample.name)
                actual = predict_maps(self.output / name / "fold_1.pt", sample, self.fixture / f"{name}_heldout.csv")[0]
                self.assertAlmostEqual(float(expected["probability_aml"]), actual["probability_aml"], delta=1e-6)
                self.assertEqual(expected["predicted_label"], actual["predicted_label"])

    def test_checkpoint_normalizer_uses_inner_train_only(self):
        checkpoint = torch.load(self.output / "dec-mil/fold_1.pt", weights_only=True)
        outer = [r for r in self.split if r["fold_id"] == "1" and r["split_role"] == "train"]
        inner, _ = inner_partition(outer, 1, self.config["seed"])
        values = []
        for row in inner:
            with np.load(self.cache / (Path(row["file_name"]).stem + ".npz")) as payload:
                values.append(payload["bag"])
        expected = np.concatenate(values).mean(axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(checkpoint["normalizer_mean"].numpy(), expected)

    def test_group_leakage_is_rejected(self):
        split = copy.deepcopy(self.split)
        next(r for r in split if r["fold_id"] == "1" and r["split_role"] == "train")["split_role"] = "test"
        with self.assertRaises(ValueError):
            validate_splits(split)

    def test_existing_weights_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            train_models(self.config, ["dec-mil"], self.cache, self.split_path, self.output)

    def test_inference_reads_downloaded_archive_format(self):
        archive_path = self.fixture / "maps.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(self.raw / "ALL_1.feather", arcname="ALL_1.feather")
        result = predict_maps(self.output / "ug-dec-mil/fold_1.pt", archive_path, self.fixture / "archive_predictions.csv")
        self.assertEqual(result[0]["file_name"], "ALL_1.feather")
        saved = next(r for r in read_csv(self.output / "ug-dec-mil/predictions.csv") if r["file_name"] == "ALL_1.feather")
        self.assertAlmostEqual(result[0]["probability_aml"], float(saved["probability_aml"]), delta=1e-6)

    def test_original_shape_patch_mean(self):
        x, y = np.meshgrid(np.arange(64, dtype=np.int32), np.arange(64, dtype=np.int32))
        spatial = (x + y * 64).astype(np.float32)
        columns = {str(channel): pa.array((spatial + channel).ravel()) for channel in range(1600)}
        columns.update(x_pos=pa.array(x.ravel()), y_pos=pa.array(y.ravel()))
        bag = make_bag(pa.table(columns), load_config()["preprocessing"])
        self.assertEqual(bag.shape, (64, 1600))
        self.assertEqual(bag[0, 0], spatial[:8, :8].mean())


if __name__ == "__main__":
    unittest.main()
