import glob
import numpy as np
import tensorflow as tf

# ==============================================================================
# Finds all pruned .tflite models in the current directory and reports
# measured weight sparsity for each. Works directly on the .tflite format,
# since multi_seed_experiment.py exports only .tflite files (no .h5).
# ==============================================================================
model_paths = sorted(glob.glob("*pruned*.tflite"))

if not model_paths:
    print("No pruned .tflite models found in the current directory.")
    print("Expected filenames like: MCDNN_pruned_seed42.tflite")
else:
    for path in model_paths:
        print(f"\nSparsity profile for: {path}")
        print("-" * 60)

        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        total_weights = 0
        zero_weights = 0

        # Pull tensor details and inspect weight-like tensors (2D+, not biases)
        tensor_details = interpreter.get_tensor_details()
        for detail in tensor_details:
            try:
                tensor = interpreter.get_tensor(detail['index'])
            except ValueError:
                continue  # some tensors aren't readable (e.g. dynamic shapes)

            if tensor.ndim >= 2:
                n_total = tensor.size
                n_zero = np.sum(tensor == 0)
                total_weights += n_total
                zero_weights += n_zero
                if n_total > 0:
                    pct = 100.0 * n_zero / n_total
                    name = detail['name'][:40]
                    print(f"{name:42s} shape={str(tensor.shape):16s} sparsity={pct:.2f}%")

        print("-" * 60)
        if total_weights > 0:
            overall_sparsity = 100.0 * zero_weights / total_weights
            print(f"Global model sparsity: {overall_sparsity:.2f}%")
            print(f"Total weights:         {total_weights:,}")
            print(f"Zero parameters:       {zero_weights:,}")
        else:
            print("No 2D+ weight tensors found to analyze.")
