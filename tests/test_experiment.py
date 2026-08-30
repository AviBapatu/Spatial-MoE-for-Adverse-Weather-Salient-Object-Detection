import os
import shutil
import unittest
import json
from src.config import ExperimentConfig
from src.experiment import setup_experiment_run, generate_experiment_id, get_git_identity, update_registry_status
from src.ablations import load_baseline, generate_moe_ladder

class TestExperimentFramework(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_experiments_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config = ExperimentConfig()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_config_hash_stability(self):
        # Ensure that order of modification doesn't change hash
        c1 = ExperimentConfig()
        c1.model.backbone = "test1"
        c1.train.epochs = 100
        
        c2 = ExperimentConfig()
        c2.train.epochs = 100
        c2.model.backbone = "test1"
        
        self.assertEqual(c1.get_canonical_hash(), c2.get_canonical_hash())
        
        # Ensure volatile fields are ignored
        c2.experiment_id = "random"
        c2.run_id = "random_run"
        self.assertEqual(c1.get_canonical_hash(), c2.get_canonical_hash())
        
    def test_experiment_id_generation(self):
        c = ExperimentConfig()
        c.model.backbone = "pvt_v2_b4"
        c.model.num_experts = 8
        c.model.top_k = 2
        c.opt.effective_global_batch = 32
        
        eid = generate_experiment_id(c)
        self.assertTrue("EXP_B4_E8_K2_S32" in eid)
        
    def test_run_directory_scaffold(self):
        run_dir = setup_experiment_run(self.config, base_dir=self.test_dir, dry_run=False)
        self.assertTrue(os.path.exists(run_dir))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "code_identity.json")))
        
    def test_registry_updates(self):
        run_dir = setup_experiment_run(self.config, base_dir=self.test_dir, dry_run=False)
        registry_path = os.path.join(self.test_dir, "registry.csv")
        self.assertTrue(os.path.exists(registry_path))
        
        # Verify initial state
        with open(registry_path, "r") as f:
            lines = f.readlines()
        self.assertTrue("CREATED" in lines[1])
        
        # Update
        update_registry_status(self.test_dir, self.config.run_id, "COMPLETED")
        with open(registry_path, "r") as f:
            lines = f.readlines()
        self.assertTrue("COMPLETED" in lines[1])
        self.assertTrue("CREATED" not in lines[1])
        
    def test_moe_ladder(self):
        ladder = generate_moe_ladder(self.config)
        self.assertEqual(len(ladder), 3)
        self.assertEqual(ladder[0].model.moe_type, "none")
        self.assertEqual(ladder[1].model.moe_type, "dense")
        self.assertEqual(ladder[2].model.moe_type, "sparse")

if __name__ == "__main__":
    unittest.main()
