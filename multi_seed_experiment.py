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
        results = load_results() 
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
            print("This run did not save successfully. Re-running this script")
            print("later will retry it automatically.")
            print("Continuing to the next run...\n")

print("\n\nAll runs attempted. Check multiseed_results.json / run")
print("summarize_multiseed.py to see final status.")
