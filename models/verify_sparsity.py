import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tf_keras
import glob

models_list = glob.glob("models/saved_artifacts/*pruned*.h5")
if not models_list:
    print("No pruned models found. Run train_seeds.py first.")
else:
    for model_path in models_list:
        print(f"\nVerifying Sparsity for: {os.path.basename(model_path)}")
        print("-" * 60)
        model = tf_keras.models.load_model(model_path)
        total_weights, zero_weights = 0, 0
        for layer in model.layers:
            for w in layer.get_weights():
                if w.ndim >= 2:
                    n_total = w.size
                    n_zero = np.sum(w == 0)
                    total_weights += n_total
                    zero_weights += n_zero
                    print(f"{layer.name:25s} Shape: {str(w.shape):15s} Sparsity: {100.0 * n_zero / n_total:.2f}%")
        print(f"OVERALL MODEL SPARSITY: {100.0 * zero_weights / total_weights:.2f}%")
