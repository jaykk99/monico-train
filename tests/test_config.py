"""Tests for config.py — hyperparameter validation. CPU only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TrainConfig, from_dict, validate


class TestDefaultsValid(unittest.TestCase):
    def test_defaults_pass(self):
        self.assertEqual(validate(TrainConfig()), [])

    def test_effective_batch_sane(self):
        c = TrainConfig()
        self.assertEqual(c.per_device_train_batch_size * c.gradient_accumulation_steps, 8)


class TestInvalidConfigs(unittest.TestCase):
    def test_negative_lr(self):
        c = TrainConfig(learning_rate=-1e-4)
        self.assertTrue(any("learning_rate" in p for p in validate(c)))

    def test_zero_lora_r(self):
        self.assertTrue(any("lora_r" in p for p in validate(TrainConfig(lora_r=0))))

    def test_empty_target_modules(self):
        self.assertTrue(any("target_modules" in p for p in validate(TrainConfig(target_modules=()))))

    def test_bad_optimizer(self):
        self.assertTrue(any("optim" in p for p in validate(TrainConfig(optim="fancy_adam"))))

    def test_zero_rows(self):
        c = TrainConfig(code_rows=0, chat_rows=0, identity_rows=0)
        self.assertTrue(any("0" in p for p in validate(c)))

    def test_huge_batch_warns_vram(self):
        c = TrainConfig(per_device_train_batch_size=64)
        self.assertTrue(any("VRAM" in p for p in validate(c)))

    def test_absolute_output_dir(self):
        self.assertTrue(any("output_dir" in p for p in validate(TrainConfig(output_dir="/tmp/x"))))

    def test_tiny_identity(self):
        self.assertTrue(any("identity" in p for p in validate(TrainConfig(identity_rows=5))))

    def test_bad_base_model(self):
        self.assertTrue(any("base_model" in p for p in validate(TrainConfig(base_model="nota-repo"))))


class TestFromDict(unittest.TestCase):
    def test_round_trip(self):
        c = TrainConfig(lora_r=32, num_train_epochs=2.0)
        d = {"lora_r": 32, "num_train_epochs": 2.0, "target_modules": ["q_proj"]}
        c2 = from_dict(d)
        self.assertEqual(c2.lora_r, 32)
        self.assertEqual(c2.num_train_epochs, 2.0)
        self.assertEqual(c2.target_modules, ("q_proj",))
        self.assertEqual(c2.base_model, c.base_model)  # defaults preserved

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            from_dict({"nope": 1})

    def test_cli_check_file(self):
        from config import main
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"learning_rate": 5.0}, f)
            path = f.name
        self.assertEqual(main(["--check", path]), 1)   # lr=5 way out of range
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path2 = f.name
        self.assertEqual(main(["--check", path2]), 0)  # empty = all defaults = valid


if __name__ == "__main__":
    unittest.main()
