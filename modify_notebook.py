import json

with open("fck-workkkk (1).ipynb", "r") as f:
    nb = json.load(f)

# Cell 33
cell_33_content = """if RUN_MODE == "TRAIN":
    print("Running Preflight Dry Run (2-5 steps)...")

    # Build a Kaggle runtime config from the canonical experiment config.
    # The only environment-specific change is the actual dataset root
    # discovered by the notebook.
    with open(
        os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r"
    ) as f:
        runtime_cfg = json.load(f)

    runtime_cfg["data"]["dataset_root"] = valid_root

    RUNTIME_CONFIG = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "kaggle_runtime.json"
    )

    with open(RUNTIME_CONFIG, "w") as f:
        json.dump(runtime_cfg, f, indent=4)

    print("Runtime dataset root:")
    print(runtime_cfg["data"]["dataset_root"])
    print("Runtime config:")
    print(RUNTIME_CONFIG)

    res = subprocess.run([
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "src.train_ddp",
        "--config",
        RUNTIME_CONFIG,
        "--preflight",
        "--max_optimizer_steps",
        "5",
    ], cwd=PROJECT_ROOT, check=True)

    with open(
        os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r"
    ) as f:
        pf_data = json.load(f)

    with open(
        os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r"
    ) as f:
        canonical_cfg = json.load(f)

    from src.train_ddp import get_config_hash

    canonical_hash = get_config_hash(
        canonical_cfg,
        "model_config_hash"
    )

    if (
        pf_data.get("status") == "PASS"
        and pf_data.get("config_hash") == canonical_hash
    ):
        mark_gate(
            "PREFLIGHT_DRY_RUN_CHECK",
            "PASS",
            run_id=pf_data.get("run_id"),
            config_hash=pf_data.get("config_hash"),
        )
    else:
        mark_gate("PREFLIGHT_DRY_RUN_CHECK", "FAIL")
        raise RuntimeError(
            f"Preflight validation failed: {pf_data}"
        )"""

# I should preserve FINAL_STATUS block if the user just meant the first block?
# Let's check what the user provided. "Cell 33 — replace the entire code block" 
# Actually, the user says "Cell 33 — replace the entire code block\nif RUN_MODE == 'TRAIN':...". 
# The user might have thought cell 33 was just that part. Let's just replace the whole cell to follow "entire code block". 
# Wait, if FINAL_STATUS goes missing, cell 35 will fail: `if FINAL_STATUS != "PASS" or ...`
# Oh! In the notebook, `FINAL_STATUS = "PASS"` is part of cell 33. If I delete it, cell 35 will crash.
# The user might have meant replacing the `if RUN_MODE == "TRAIN":` block only. Let me keep the rest.

cell_33_original = "".join(nb["cells"][33]["source"])
import re
new_cell_33 = re.sub(
    r'^if RUN_MODE == "TRAIN":.*?mark_gate\("PREFLIGHT_DRY_RUN_CHECK", "FAIL"\)',
    cell_33_content,
    cell_33_original,
    flags=re.DOTALL | re.MULTILINE
)

# If it failed to replace, it means it didn't match. 
if new_cell_33 == cell_33_original:
    print("Warning: regex for cell 33 didn't match. Replacing the whole cell just in case.")
    new_cell_33 = cell_33_content + "\n\n" + cell_33_original[cell_33_original.find("FINAL_STATUS = \"PASS\""):]
    
nb["cells"][33]["source"] = [line + ("\n" if i < len(new_cell_33.split("\n"))-1 else "") for i, line in enumerate(new_cell_33.split("\n"))]

# Cell 35
cell_35_original = "".join(nb["cells"][35]["source"])
cell_35_replacement = """print(f"Validated config hash: {canonical_hash}")
print(f"Training config hash:  {actual_hash}")

if canonical_hash != actual_hash:
    raise RuntimeError(
        "Config hash mismatch between canonical config and preflight runtime!"
    )

print("MATCH")
print("Launching final 50-epoch training...")

subprocess.run([
    "torchrun",
    "--nproc_per_node=2",
    "-m",
    "src.train_ddp",
    "--config",
    RUNTIME_CONFIG,
], cwd=PROJECT_ROOT, check=True)"""

new_cell_35 = re.sub(
    r'print\(f"Validated config hash: \{canonical_hash\}"\).*?check=True\)',
    cell_35_replacement,
    cell_35_original,
    flags=re.DOTALL
)
nb["cells"][35]["source"] = [line + ("\n" if i < len(new_cell_35.split("\n"))-1 else "") for i, line in enumerate(new_cell_35.split("\n"))]

# Cell 39
cell_39_original = "".join(nb["cells"][39]["source"])

cell_39_add_top = """with open(
    os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r"
) as f:
    runtime_cfg = json.load(f)

runtime_cfg["data"]["dataset_root"] = valid_root

RUNTIME_CONFIG = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "kaggle_runtime.json"
)

with open(RUNTIME_CONFIG, "w") as f:
    json.dump(runtime_cfg, f, indent=4)

"""

new_cell_39 = cell_39_original.replace(
    'subprocess.run([\n        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp", "--resume", "latest", "--preflight", "--max_optimizer_steps", "5", "--data_root", valid_root\n    ], cwd=PROJECT_ROOT, check=True)',
    '''subprocess.run([
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "src.train_ddp",
        "--config",
        RUNTIME_CONFIG,
        "--resume",
        "latest",
        "--preflight",
        "--max_optimizer_steps",
        "5",
    ], cwd=PROJECT_ROOT, check=True)'''
).replace(
    'subprocess.run([\n    "torchrun",\n    "--nproc_per_node=2",\n    "-m",\n    "src.train_ddp",\n    "--resume",\n    "latest",\n    "--data_root",\n    valid_root,\n    ], cwd=PROJECT_ROOT, check=True)',
    '''subprocess.run([
    "torchrun",
    "--nproc_per_node=2",
    "-m",
    "src.train_ddp",
    "--config",
    RUNTIME_CONFIG,
    "--resume",
    "latest",
], cwd=PROJECT_ROOT, check=True)'''
)

new_cell_39_lines = new_cell_39.split('\n')
new_cell_39_lines.insert(1, "\n".join(["    " + line for line in cell_39_add_top.split("\n")])) # Add inside if RUN_MODE == "RESUME":
new_cell_39 = "\n".join(new_cell_39_lines)

nb["cells"][39]["source"] = [line + ("\n" if i < len(new_cell_39.split("\n"))-1 else "") for i, line in enumerate(new_cell_39.split("\n"))]

with open("fck-workkkk (1).ipynb", "w") as f:
    json.dump(nb, f, indent=1)

