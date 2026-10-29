"""Failed runs preserve outputs and give actionable restart messages."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import preprocess
import train
from src.data import load_bags, load_config, prepare_data, write_csv


class InterruptedRunTests(unittest.TestCase):
    def setUp(self):
        generated = ROOT / "outputs"
        generated.mkdir(exist_ok=True)
        self.fixture = Path(tempfile.mkdtemp(prefix="test_interrupted_", dir=generated))
        self.manifest = self.fixture / "manifest.csv"
        write_csv(self.manifest, [{"file_name": "first.feather"}, {"file_name": "second.feather"}])

    def test_preprocessing_failure_and_ctrl_c_preserve_partial_cache(self):
        for index, error in enumerate([ValueError("Broken map"), KeyboardInterrupt()]):
            with self.subTest(error=type(error).__name__):
                cache = self.fixture / f"cache_{index}"
                with mock.patch("src.data.iter_maps", return_value=iter([("first.feather", None), ("second.feather", None)])), \
                     mock.patch("src.data.make_bag", side_effect=[np.zeros((64, 16), dtype=np.float32), error]):
                    with self.assertRaises(type(error)) as caught:
                        prepare_data(self.fixture, cache, self.manifest, {})
                self.assertIs(caught.exception, error)
                self.assertTrue((cache / "first.npz").is_file())
                self.assertFalse((cache / "preprocessing.json").exists())
                note = " ".join(error.__notes__)
                self.assertIn(str(cache.resolve()), note)
                self.assertIn("1/2 maps were saved", note)
                self.assertIn("paths.cache", note)
                self.assertIn("from the beginning", note)

    def test_existing_cache_has_safe_rebuild_instructions(self):
        cache = self.fixture / "existing_cache"
        cache.mkdir()
        sentinel = cache / "keep.txt"
        sentinel.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError) as caught:
            prepare_data(self.fixture, cache, self.manifest, {})
        self.assertIn("completed or partial", str(caught.exception))
        self.assertIn("rename", str(caught.exception))
        self.assertIn("paths.cache", str(caught.exception))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")

    def test_missing_incomplete_and_invalid_cache_messages(self):
        cache = self.fixture / "cache"
        with self.assertRaisesRegex(FileNotFoundError, "Run python preprocess.py first"):
            load_bags(cache, [], {})
        cache.mkdir()
        with self.assertRaisesRegex(ValueError, "Cache is incomplete"):
            load_bags(cache, [], {})
        (cache / "index.csv").write_text("file_name\n", encoding="utf-8")
        (cache / "preprocessing.json").write_text("{", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError) as caught:
            load_bags(cache, [], {})
        self.assertIn("paths.cache", " ".join(caught.exception.__notes__))

    def test_training_failure_and_ctrl_c_preserve_previous_checkpoint(self):
        config = {"models": ["dec-mil"], "model_overrides": {}, "device": "cpu", "preprocessing": {}}
        for index, error in enumerate([ValueError("Nonfinite training loss"), KeyboardInterrupt()]):
            with self.subTest(error=type(error).__name__):
                output = self.fixture / f"training_{index}"

                def stop_in_second_fold(name, rows, bags, config, fold, device, model_dir):
                    if fold == 2:
                        raise error
                    (model_dir / "fold_1.pt").write_bytes(b"preserve this checkpoint")
                    return [], {"fold": 1, "balanced_accuracy": 0.5, "macro_f1": 0.5}

                with mock.patch.object(train, "read_csv", return_value=[]), \
                     mock.patch.object(train, "validate_splits"), \
                     mock.patch.object(train, "load_bags", return_value={}), \
                     mock.patch.object(train, "train_fold", side_effect=stop_in_second_fold):
                    with self.assertRaises(type(error)) as caught:
                        train.train_models(config, ["dec-mil"], self.fixture, self.manifest, output)
                    self.assertIs(caught.exception, error)
                    with self.assertRaises(FileExistsError) as retry:
                        train.train_models(config, ["dec-mil"], self.fixture, self.manifest, output)
                checkpoint = output / "dec-mil/fold_1.pt"
                self.assertEqual(checkpoint.read_bytes(), b"preserve this checkpoint")
                note = " ".join(error.__notes__)
                self.assertIn("fold 2", note)
                self.assertIn(str(checkpoint.parent.resolve()), note)
                self.assertIn("--model dec-mil", note)
                self.assertIn("paths.outputs", note)
                self.assertIn("do not resume automatically", note)
                self.assertIn("rename", str(retry.exception))

    def test_training_report_failure_identifies_final_writing_stage(self):
        config = {"models": ["dec-mil"], "model_overrides": {}, "device": "cpu", "preprocessing": {}}
        with mock.patch.object(train, "read_csv", return_value=[]), \
             mock.patch.object(train, "validate_splits"), \
             mock.patch.object(train, "load_bags", return_value={}), \
             mock.patch.object(train, "train_fold", return_value=([], {})), \
             mock.patch.object(train, "write_csv", side_effect=OSError("Disk full")):
            with self.assertRaises(OSError) as caught:
                train.train_models(config, ["dec-mil"], self.fixture, self.manifest, self.fixture / "training")
        self.assertIn("writing final results", " ".join(caught.exception.__notes__))
        self.assertEqual(str(caught.exception), "Disk full")

    def test_cli_errors_print_reason_notes_and_correct_exit_code(self):
        for module, operation in [(preprocess, "prepare_data"), (train, "train_models")]:
            for error, expected_code in [(ValueError("Broken input"), 1), (KeyboardInterrupt(), 130)]:
                with self.subTest(command=module.__name__, error=type(error).__name__):
                    error.add_note("Partial folder preserved; rename it before restarting.")
                    stderr = io.StringIO()
                    with mock.patch("sys.argv", [module.__name__ + ".py"]), \
                         mock.patch.object(module, "load_config", return_value=load_config()), \
                         mock.patch.object(module, operation, side_effect=error), \
                         contextlib.redirect_stderr(stderr):
                        code = module.main()
                    self.assertEqual(code, expected_code)
                    message = stderr.getvalue()
                    self.assertIn("Partial folder preserved", message)
                    self.assertNotIn("Traceback", message)
                    self.assertIn("interrupted" if expected_code == 130 else "Broken input", message)


if __name__ == "__main__":
    unittest.main()
