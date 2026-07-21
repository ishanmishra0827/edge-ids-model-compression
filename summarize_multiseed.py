"""
Summarizes multiseed_results.json into mean +/- std for accuracy,
macro F1, weighted F1, and latency, for both dense and pruned models.
Works even if only some seeds have finished -- it reports how many
are done and averages over whatever's available.

Run: python summarize_multiseed.py
"""
import json
import numpy as np

with open("multiseed_results.json", "r") as f:
    results = json.load(f)

for variant in ["dense", "pruned"]:
    runs = [v for k, v in results.items() if v["variant"] == variant]
    n = len(runs)
    print(f"\n{'='*60}")
    print(f"  {variant.upper()} MODEL — {n}/5 seeds completed")
    print(f"{'='*60}")
    if n == 0:
        print("No runs completed yet.")
        continue

    for metric in ["accuracy", "macro_f1", "weighted_f1", "latency_ms"]:
        values = np.array([r[metric] for r in runs])
        mean = values.mean()
        std = values.std(ddof=1) if n > 1 else 0.0
        print(f"{metric:15s}: {mean:.4f} +/- {std:.4f}   (values: {[round(v,4) for v in values]})")

print(f"\n{'='*60}")
print("LaTeX-ready table row format (once all 5 seeds per variant are done):")
print(f"{'='*60}")
for variant in ["dense", "pruned"]:
    runs = [v for k, v in results.items() if v["variant"] == variant]
    if len(runs) < 5:
        print(f"{variant}: incomplete ({len(runs)}/5 seeds) — not ready yet")
        continue
    acc = np.array([r["accuracy"] for r in runs])
    f1 = np.array([r["macro_f1"] for r in runs])
    wf1 = np.array([r["weighted_f1"] for r in runs])
    lat = np.array([r["latency_ms"] for r in runs])
    print(f"\n{variant.capitalize()}:")
    print(f"  Accuracy:     {acc.mean()*100:.1f} $\\pm$ {acc.std(ddof=1)*100:.1f}\\%")
    print(f"  Macro F1:     {f1.mean():.2f} $\\pm$ {f1.std(ddof=1):.2f}")
    print(f"  Weighted F1:  {wf1.mean():.2f} $\\pm$ {wf1.std(ddof=1):.2f}")
    print(f"  Latency:      ${lat.mean():.4f} \\pm {lat.std(ddof=1):.4f}$ ms")
