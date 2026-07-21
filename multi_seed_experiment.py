"""
MULTI-SEED VARIANCE EXPERIMENT (subprocess-based, memory-safe)
================================================================
Launches EACH of the 10 runs (5 seeds x dense/pruned) as a completely
fresh, separate Python process via subprocess. This guarantees the
operating system fully reclaims all memory between runs -- fixing the
repeated out-of-memory kills that occurred when everything ran inside
one long-lived process on this Pi's 2GB RAM.

Resumable: checks multiseed_results.json before each run and skips
anything already completed. Safe to stop (Ctrl+C) and restart anytime.

Run: python3 multi_seed_experiment.py
"""
import json
import os
import subprocess
import sys

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_FILE = "multiseed_results.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}


for seed in SEEDS:
    for variant in ["dense", "pruned"]:
        key = f"{variant}_seed{seed}"
        results = load_results()  # re-check fresh each time in case of manual edits
        if key in results:
            print(f"Skipping {key} (already completed)")
            continue

        print(f"\n{'#'*60}")
        print(f"# Launching fresh process for: {key}")
        print(f"{'#'*60}")

        result = subprocess.run(
            [sys.executable, "run_single_seed.py", str(seed), variant]
        )

        if result.returncode != 0:
            print(f"\nWARNING: {key} exited with error code {result.returncode}.")
            print("This run did NOT save successfully. Re-running this script")
            print("later will retry it automatically (it won't be in the")
            print("results file yet, so it won't be skipped).")
            print("Continuing to the next run...\n")

print("\n\nALL RUNS ATTEMPTED. Check multiseed_results.json / run")
print("summarize_multiseed.py to see final status.")
