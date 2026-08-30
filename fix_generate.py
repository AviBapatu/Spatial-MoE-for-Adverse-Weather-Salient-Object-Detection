with open("generate_notebook.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "canonical_hash = get_config_hash(canonical_cfg," in line:
        print(f"Found at line {i}: {line}")
