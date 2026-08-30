import json

with open("fck-workkkk (1).ipynb", "r") as f:
    nb = json.load(f)

cell_35_replacement = """if RUN_MODE == "TRAIN":
    if FINAL_STATUS != "PASS" or GATES.get("PREFLIGHT_DRY_RUN_CHECK") != "PASS":
        raise RuntimeError("Refusing to train: Not all gates passed.")

    with open(
        os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r"
    ) as f:
        canonical_cfg = json.load(f)

    from src.train_ddp import get_config_hash

    canonical_hash = get_config_hash(
        canonical_cfg,
        "model_config_hash"
    )

    with open(
        os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r"
    ) as f:
        pf_data = json.load(f)

    actual_hash = pf_data.get("config_hash")

    print(f"Validated config hash: {canonical_hash}")
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

nb["cells"][35]["source"] = [line + ("\n" if i < len(cell_35_replacement.split("\n"))-1 else "") for i, line in enumerate(cell_35_replacement.split("\n"))]

with open("fck-workkkk (1).ipynb", "w") as f:
    json.dump(nb, f, indent=1)

