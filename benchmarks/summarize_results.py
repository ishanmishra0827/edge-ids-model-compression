import json
import numpy as np
import os

RESULTS_FILE = "benchmarks/multiseed_results.json"
if not os.path.exists(RESULTS_FILE):
    print("No telemetry file located. Complete training cycles first.")
else:
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    for variant in ["dense", "pruned"]:
        runs = [v for k, v in results.items() if v["variant"] == variant]
        n = len(runs)
        print(f"\n=== {variant.upper()} EXPERIMENTAL MATRIX ({n}/5 Complete) ===")
        if n == 0: continue
        for metric in ["accuracy", "macro_f1", "weighted_f1", "latency_ms"]:
            values = np.array([r[metric] for r in runs])
            std = values.std(ddof=1) if n > 1 else 0.0
            print(f"{metric:15s}: {values.mean():.4f} +/- {std:.4f}")
