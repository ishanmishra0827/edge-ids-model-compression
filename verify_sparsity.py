import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras

model = tf_keras.models.load_model("MCDNN_structured_pruned.h5")

total_weights = 0
zero_weights = 0

print("Per-layer sparsity:")
print("-" * 60)
for layer in model.layers:
    weights = layer.get_weights()
    for w in weights:
        if w.ndim >= 2:
            n_total = w.size
            n_zero = np.sum(w == 0)
            total_weights += n_total
            zero_weights += n_zero
            if n_total > 0:
                pct = 100.0 * n_zero / n_total
                print(f"{layer.name:30s} shape={str(w.shape):20s} sparsity={pct:.2f}%")

print("-" * 60)
overall_sparsity = 100.0 * zero_weights / total_weights
print(f"\nOVERALL MODEL SPARSITY: {overall_sparsity:.2f}%")
print(f"Total weight parameters: {total_weights:,}")
print(f"Zero-valued parameters:  {zero_weights:,}")
print(f"Non-zero parameters:     {total_weights - zero_weights:,}")
