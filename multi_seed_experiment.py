"""
MULTI-SEED VARIANCE EXPERIMENT
================================
Trains BOTH the dense baseline and the pruned model across 5 random
seeds each (10 total training runs), evaluates each on the real
KDDTest+ known-class subset, and benchmarks latency the same way as
before. Results are saved incrementally to 'multiseed_results.json'
after EVERY run, so if you need to stop and restart, nothing is lost
-- it will skip runs already completed.

REALISTIC TIME ESTIMATE: ~15-16 min per run x 10 runs = ~2.5-3 hours
total. You can safely stop this (Ctrl+C) and resume later; it picks
up where it left off.

Run: python multi_seed_experiment.py
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import time
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras
import scipy.stats as stats
from tf_keras import layers
from sklearn.metrics import classification_report

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_FILE = "multiseed_results.json"

# ==============================================================================
# LOAD DATA ONCE (same fixed preprocessed data for every seed/run --
# only the model's weight initialization and training stochasticity vary,
# which is the standard approach for seed-variance studies)
# ==============================================================================
x_train = np.load("x_train.npy").astype(np.float32)
y_train = np.load("y_train.npy").astype(np.float32)
x_val = np.load("x_val.npy").astype(np.float32)
y_val = np.load("y_val.npy").astype(np.float32)
x_test_real = np.load("x_test_real.npy").astype(np.float32)
y_test_real = np.load("y_test_real.npy")
y_test_ints = np.argmax(y_test_real, axis=1)
num_classes = y_train.shape[1]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def build_model():
    inputs = layers.Input(shape=(x_train.shape[1],), name="network_input")
    x1 = layers.Dense(128, activation='relu', name='dense_block1')(inputs)
    x1_drop = layers.Dropout(0.2)(x1)
    head1 = layers.Dense(num_classes, activation='softmax', name='head1')(x1_drop)
    x2 = layers.Dense(256, activation='relu', name='dense_block2')(x1_drop)
    x2_drop = layers.Dropout(0.2)(x2)
    head2 = layers.Dense(num_classes, activation='softmax', name='head2')(x2_drop)
    x3 = layers.Dense(128, activation='relu', name='dense_block3')(x2_drop)
    head3 = layers.Dense(num_classes, activation='softmax', name='head3')(x3)
    return tf_keras.Model(inputs=inputs, outputs=[head1, head2, head3])


def profile_tflite_latency(model_path, num_trials=10):
    test_samples = x_test_real[:1000]
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    trial_means = []
    for t in range(num_trials):
        start_time = time.perf_counter()
        for sample in test_samples:
            input_data = sample.reshape(1, -1)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        trial_means.append(total_time_ms / len(test_samples))
    trial_means = np.array(trial_means)
    return float(np.mean(trial_means)), trial_means.tolist()


def evaluate_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    try:
        head3_index = next(d['index'] for d in output_details if 'head3' in d['name'].lower())
    except StopIteration:
        head3_index = output_details[-1]['index']

    y_pred = []
    for sample in x_test_real:
        interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
        interpreter.invoke()
        y_pred.append(np.argmax(interpreter.get_tensor(head3_index)))

    report = classification_report(y_test_ints, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def run_one(seed, use_pruning):
    variant = "pruned" if use_pruning else "dense"
    key = f"{variant}_seed{seed}"

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model()

    if use_pruning:
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.7453, begin_step=0),
            'block_size': (1, 4),
            'block_pooling_type': 'AVG'
        }
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        fit_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    else:
        fit_callbacks = []

    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        metrics=['accuracy'],
        loss_weights=[0.2, 0.3, 0.5]
    )

    print(f"\n{'='*60}\nTraining {key}...\n{'='*60}")
    model.fit(
        x_train, [y_train, y_train, y_train],
        epochs=15,
        batch_size=128,
        validation_data=(x_val, [y_val, y_val, y_val]),
        callbacks=fit_callbacks,
        verbose=1
    )

    if use_pruning:
        model = tfmot.sparsity.keras.strip_pruning(model)

    tflite_path = f"MCDNN_{key}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Evaluating {key} on real test set...")
    metrics = evaluate_tflite(tflite_path)

    print(f"Profiling latency for {key}...")
    latency_mean, trial_means = profile_tflite_latency(tflite_path)

    result = {
        "seed": seed,
        "variant": variant,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "latency_ms": latency_mean,
        "latency_trials_ms": trial_means,
    }
    print(f"Result for {key}: {result}")
    return key, result


# ==============================================================================
# MAIN LOOP -- resumable
# ==============================================================================
results = load_results()

for seed in SEEDS:
    for use_pruning in [False, True]:
        variant = "pruned" if use_pruning else "dense"
        key = f"{variant}_seed{seed}"
        if key in results:
            print(f"Skipping {key} (already completed)")
            continue
        key, result = run_one(seed, use_pruning)
        results[key] = result
        save_results(results)  # save after EVERY run

print("\n\nALL RUNS COMPLETE. Results saved to multiseed_results.json")
print("Run summarize_multiseed.py next to compute mean +/- std.")
