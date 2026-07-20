import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras
import glob

# Dynamically find any pruned H5 file in your artifacts folder
model_paths = glob.glob("models/saved_artifacts/*pruned*.h5")

if not model_paths:
    print("No pruned models found in models/saved_artifacts/")
else:
    for path in model_paths:
        print(f"\nSparsity profile for: {os.path.basename(path)}")
        print("-" * 60)
        
        model = tf_keras.models.load_model(path)
        total_weights = 0
        zero_weights = 0

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
                        print(f"{layer.name:25s} shape={str(w.shape):18s} layer_sparsity={pct:.2f}%")

        print("-" * 60)
        overall_sparsity = 100.0 * zero_weights / total_weights
        print(f"Global model sparsity: {overall_sparsity:.2f}%")
        print(f"Total weights:         {total_weights:,}")
        print(f"Zero parameters:       {zero_weights:,}")
