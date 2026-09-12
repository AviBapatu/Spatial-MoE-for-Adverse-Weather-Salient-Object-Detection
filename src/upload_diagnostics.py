"""Upload diagnostic JSON artifacts (routing stats, expert similarities,
padding audit) to the ``diagnostics/`` folder of the HF dataset repo.

Skips files that don't exist locally — used after a diagnostics run on a
notebook/session that produced evaluation artifacts.
"""

import os
from typing import List, Tuple

from src.hf_sync import push_json

FILES_TO_UPLOAD: List[Tuple[str, str]] = [
    ("evaluation/diagnostics_test_real/routing_stats_ep0.json", "diagnostics/routing_stats_test_real.json"),
    ("evaluation/diagnostics_test_sys/routing_stats_ep0.json", "diagnostics/routing_stats_test_sys.json"),
    ("evaluation/diagnostics_test_real/expert_similarity_moe_4.json", "diagnostics/expert_similarity_test_real_moe_4.json"),
    ("evaluation/diagnostics_test_real/expert_similarity_moe_8.json", "diagnostics/expert_similarity_test_real_moe_8.json"),
    ("evaluation/diagnostics_test_real/expert_similarity_moe_16.json", "diagnostics/expert_similarity_test_real_moe_16.json"),
    ("evaluation/diagnostics_test_sys/expert_similarity_moe_4.json", "diagnostics/expert_similarity_test_sys_moe_4.json"),
    ("evaluation/diagnostics_test_sys/expert_similarity_moe_8.json", "diagnostics/expert_similarity_test_sys_moe_8.json"),
    ("evaluation/diagnostics_test_sys/expert_similarity_moe_16.json", "diagnostics/expert_similarity_test_sys_moe_16.json"),
    ("evaluation/padding_stats.json", "diagnostics/padding_stats.json"),
]


def upload() -> None:
    """Push every existing file in :data:`FILES_TO_UPLOAD` to the Hub."""
    for local_path, remote_name in FILES_TO_UPLOAD:
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {remote_name}...")
            push_json(local_path, name=remote_name)
        else:
            print(f"Warning: File {local_path} does not exist. Skipping.")


if __name__ == "__main__":
    upload()
