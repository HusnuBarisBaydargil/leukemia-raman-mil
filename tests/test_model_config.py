"""Configuration changes must affect models and survive checkpoint loading."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import load_checkpoint
from src.data import load_config
from src.models import build_model, probabilities


class ModelConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load_config()
        cls.config["model"].update(instance_hidden_dim=40, embedding_dim=24, dropout=0.35)
        cls.config["model"]["heads"] = {
            "dec-mil": {"attention_hidden_dim": 11, "top_k": 3},
            "ug-dec-mil": {"attention_hidden_dim": 11, "top_k": 3},
            "attention-mil": {"attention_hidden_dim": 12},
            "abmil": {"attention_hidden_dim": 13},
            "clam-sb": {"projection_dim": 40, "attention_hidden_dim": 17,
                        "projection_dropout": 0.1, "k_sample": 4},
        }
        cls.inputs = torch.linspace(-1, 1, 2 * 64 * 16).reshape(2, 64, 16)

    def test_shared_encoder_settings_apply_to_all_models(self):
        for name in self.config["models"]:
            with self.subTest(model=name):
                model = build_model(name, 16, self.config["model"])
                encoder = model.encoder if name in {"abmil", "clam-sb"} else model
                self.assertEqual(encoder.instance_encoder[0].out_features, 40)
                self.assertEqual(encoder.instance_encoder[2].p, 0.35)
                self.assertEqual(encoder.instance_encoder[3].out_features, 24)
                model.eval()
                probability, prediction = probabilities(model, name, self.inputs)
                self.assertEqual(tuple(probability.shape), (2,))
                self.assertTrue(torch.isfinite(probability).all())
                self.assertEqual(tuple(prediction.shape), (2,))

    def test_model_specific_head_settings(self):
        models = {name: build_model(name, 16, self.config["model"]) for name in self.config["models"]}
        for name in ("dec-mil", "ug-dec-mil"):
            self.assertEqual(models[name].attention[0].out_features, 11)
            self.assertEqual(models[name](self.inputs)[-1].shape[-1], 3)
        self.assertEqual(models["attention-mil"].attention[0].out_features, 12)
        self.assertEqual(models["abmil"].attn_v.out_features, 13)
        clam = models["clam-sb"]
        self.assertEqual(clam.proj[0].out_features, 40)
        self.assertEqual(clam.proj[2].p, 0.1)
        self.assertEqual(clam.attention.attn_a[0].out_features, 17)
        self.assertEqual(clam.k_sample, 4)
        outputs = clam(self.inputs, torch.tensor([0, 1]))
        outputs["instance_loss"].backward()
        self.assertTrue(torch.isfinite(outputs["instance_loss"]))

    def test_custom_configuration_checkpoint_round_trip(self):
        generated = ROOT / "outputs"
        generated.mkdir(exist_ok=True)
        fixture = Path(tempfile.mkdtemp(prefix="test_model_config_", dir=generated))
        for name in self.config["models"]:
            with self.subTest(model=name):
                model = build_model(name, 16, self.config["model"]).eval()
                with torch.no_grad():
                    expected_probability, expected_prediction = probabilities(model, name, self.inputs)
                path = fixture / f"{name}.pt"
                torch.save({"format_version": 1, "classes": {"ALL": 0, "AML": 1},
                            "model_name": name, "input_channels": 16, "config": self.config,
                            "state_dict": model.state_dict()}, path)
                loaded, _ = load_checkpoint(path)
                with torch.no_grad():
                    actual_probability, actual_prediction = probabilities(loaded, name, self.inputs)
                torch.testing.assert_close(actual_probability, expected_probability, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual_prediction, expected_prediction))

    def test_legacy_default_settings_still_load(self):
        current = load_config()
        legacy = copy.deepcopy(current["model"])
        legacy.pop("heads")
        legacy.update(attention_hidden_dim=32, top_k=8)
        for name in current["models"]:
            with self.subTest(model=name):
                torch.manual_seed(17)
                old_format = build_model(name, 16, legacy).eval()
                torch.manual_seed(17)
                new_format = build_model(name, 16, current["model"]).eval()
                old_state, new_state = old_format.state_dict(), new_format.state_dict()
                self.assertEqual(old_state.keys(), new_state.keys())
                for key in old_state:
                    self.assertTrue(torch.equal(old_state[key], new_state[key]), key)


if __name__ == "__main__":
    unittest.main()
