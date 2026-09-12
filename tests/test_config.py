"""Tests for src.config: save/load round-trip, canonical hash, validation, presets."""
import json
import os
import shutil
import tempfile
import unittest

from src.config import ExperimentConfig, list_presets


class TestConfigRoundTrip(unittest.TestCase):
    """Round-trip save/load should produce identical configs."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_load_round_trip(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.backbone = "pvt_v2_b2"
        cfg.model.num_experts = 12
        cfg.model.top_k = 3
        cfg.loss.boundary_weight = 0.5
        cfg.opt.batch_per_gpu = 2

        path = os.path.join(self.tmpdir, "config.json")
        cfg.save(path)
        loaded = ExperimentConfig.load(path)

        self.assertEqual(cfg.to_dict(), loaded.to_dict())

    def test_round_trip_preserves_volatile_fields(self) -> None:
        cfg = ExperimentConfig()
        cfg.experiment_id = "EXP_TEST"
        cfg.run_id = "EXP_TEST__20260101_000000_abc"
        cfg.batch_equivalence = "NON_MATCHED"

        path = os.path.join(self.tmpdir, "config.json")
        cfg.save(path)
        loaded = ExperimentConfig.load(path)

        self.assertEqual(loaded.experiment_id, "EXP_TEST")
        self.assertEqual(loaded.batch_equivalence, "NON_MATCHED")

    def test_from_dict_ignores_unknown_keys(self) -> None:
        data = ExperimentConfig().to_dict()
        data["future_field"] = "surprise"
        data["model"]["future_model_field"] = 42

        cfg = ExperimentConfig.from_dict(data)
        self.assertEqual(cfg.model.num_experts, 8)


class TestCanonicalHash(unittest.TestCase):
    """Hash should be stable and respect volatile-key exclusions."""

    def test_same_config_same_hash(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        self.assertEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_modification_order_independent(self) -> None:
        a = ExperimentConfig()
        a.model.backbone = "test1"
        a.train.epochs = 100

        b = ExperimentConfig()
        b.train.epochs = 100
        b.model.backbone = "test1"

        self.assertEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_volatile_fields_ignored(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        b.experiment_id = "EXP_RANDOM"
        b.run_id = "EXP_RANDOM__20260101_abc"
        b.batch_equivalence = "NON_MATCHED"

        self.assertEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_dataset_root_ignored(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        b.data.dataset_root = "/completely/different/path"

        self.assertEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_num_experts_changes_hash(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        b.model.num_experts = 12

        self.assertNotEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_top_k_changes_hash(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        b.model.top_k = 3

        self.assertNotEqual(a.get_canonical_hash(), b.get_canonical_hash())

    def test_loss_weight_changes_hash(self) -> None:
        a = ExperimentConfig()
        b = ExperimentConfig()
        b.loss.ssim_weight = 1.0

        self.assertNotEqual(a.get_canonical_hash(), b.get_canonical_hash())


class TestValidation(unittest.TestCase):
    """validate() should raise ValueError on invalid combos."""

    def test_valid_config_passes(self) -> None:
        cfg = ExperimentConfig()
        cfg.validate()  # should not raise

    def test_top_k_exceeds_num_experts(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.top_k = 10
        cfg.model.num_experts = 4
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("top_k", str(ctx.exception))

    def test_invalid_moe_type(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.moe_type = "ultra_sparse"
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("moe_type", str(ctx.exception))

    def test_num_experts_zero(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.num_experts = 0
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("num_experts", str(ctx.exception))

    def test_negative_loss_weight(self) -> None:
        cfg = ExperimentConfig()
        cfg.loss.bce_weight = -1.0
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("bce_weight", str(ctx.exception))

    def test_warmup_ratio_out_of_range(self) -> None:
        cfg = ExperimentConfig()
        cfg.opt.warmup_ratio = 1.5
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("warmup_ratio", str(ctx.exception))

    def test_batch_mismatch_warns(self) -> None:
        cfg = ExperimentConfig()
        cfg.opt.batch_per_gpu = 3
        cfg.opt.grad_accum_steps = 5
        cfg.opt.effective_global_batch = 32  # not divisible by 15
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg.validate()
            batch_warnings = [x for x in w if "effective_global_batch" in str(x.message)]
            self.assertEqual(len(batch_warnings), 1)

    def test_multiple_errors_reported(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.top_k = 100
        cfg.model.num_experts = 2
        cfg.model.moe_type = "bad"
        cfg.loss.bce_weight = -5.0
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        msg = str(ctx.exception)
        self.assertIn("top_k", msg)
        self.assertIn("moe_type", msg)
        self.assertIn("bce_weight", msg)


class TestPresets(unittest.TestCase):
    """from_preset() should load baseline and apply overrides."""

    def test_list_presets_non_empty(self) -> None:
        presets = list_presets()
        self.assertIn("baseline", presets)

    def test_from_preset_baseline(self) -> None:
        cfg = ExperimentConfig.from_preset("baseline")
        self.assertEqual(cfg.model.backbone, "pvt_v2_b4")
        self.assertEqual(cfg.model.num_experts, 8)
        self.assertEqual(cfg.model.top_k, 2)
        self.assertEqual(cfg.model.moe_type, "sparse")

    def test_from_preset_with_overrides(self) -> None:
        cfg = ExperimentConfig.from_preset(
            "baseline", overrides={"model.num_experts": 4}
        )
        self.assertEqual(cfg.model.num_experts, 4)
        self.assertEqual(cfg.model.backbone, "pvt_v2_b4")

    def test_from_preset_unknown_name(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ExperimentConfig.from_preset("nonexistent_preset")

    def test_preset_loads_valid_config(self) -> None:
        cfg = ExperimentConfig.from_preset("baseline")
        cfg.validate()  # should not raise


if __name__ == "__main__":
    unittest.main()
