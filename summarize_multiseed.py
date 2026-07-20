import json
import numpy as np

# Updated path matching the benchmarks folder config
with open("multiseed_results.json", "r") as f:
    results = json.load(f)

for variant in ["dense", "pruned"]:
    runs = [v for k, v in results.items() if v["variant"] == variant]
    n = len(runs)
    print(f"\n--- Matrix Analysis: {variant.upper()} ({n}/5 Complete) ---")
    if n == 0:
        print("Telemetry array payload empty.")
        continue

    for metric in ["accuracy", "macro_f1", "weighted_f1", "latency_ms"]:
        values = np.array([r[metric] for r in runs])
        mean = values.mean()
        std = values.std(ddof=1) if n > 1 else 0.0
        print(f"{metric:15s}: {mean:.4f} +/- {std:.4f}")

print("\n--- LaTeX Table Row Block Generation ---")
for variant in ["dense", "pruned"]:
    runs = [v for k, v in results.items() if v["variant"] == variant]
    if len(runs) < 5:
        print(f"{variant} state: pending final seed arrays ({len(runs)}/5 complete)")
        continue
    acc = np.array([r["accuracy"] for r in runs])
    f1 = np.array([r["macro_f1"] for r in runs])
    wf1 = np.array([r["weighted_f1"] for r in runs])
    lat = np.array([r["latency_ms"] for r in runs])
    
    print(f"\n{variant.capitalize()} Formatting:")
    print(f"  Accuracy:     {acc.mean()*100:.1f} $\\pm$ {acc.std(ddof=1)*100:.1f}\\%")
    print(f"  Macro F1:     {f1.mean():.2f} $\\pm$ {f1.std(ddof=1):.2f}")
    print(f"  Weighted F1:  {wf1.mean():.2f} $\\pm$ {wf1.std(ddof=1):.2f}")
    print(f"  Latency:      ${lat.mean():.4f} \\pm {lat.std(ddof=1):.4f}$ ms")
