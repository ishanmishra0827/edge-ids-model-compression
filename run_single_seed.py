import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import json
import time
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras
from tf_keras import layers
from sklearn.metrics import classification_report

RESULTS_FILE = "multiseed_results.json"

seed = int(sys.argv[1])
variant = sys.argv[2]
use_pruning = (variant == "pruned")
key = f"{variant}_seed{seed}"

x_train = np.load("x_train.npy").astype(np.float32)
y_train = np.load("y_train.npy").astype(np.float32)
x_val = np.load("x_val.npy").astype(np.float32)
y_val = np.load("y_val.npy").astype(np.float32)
x_test_real = np.load("x_test_real.npy").astype(np.float32)
y_test_real = np.load("y_test_real.npy")
y_test_ints = np.argmax(y_test_real, axis=1)
num_classes = y_train.shape[1]

tf.random.set_seed(seed)
np.random.seed(seed)


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
interpreter = tf.lite.Interpreter(model_path=tflite_path)
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

print(f"Profiling latency for {key}...")
test_samples = x_test_real[:1000]
trial_means = []
for t in range(10):
    start_time = time.perf_counter()
    for sample in test_samples:
        interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
        interpreter.invoke()
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    trial_means.append(total_time_ms / len(test_samples))
trial_means = np.array(trial_means)

result = {
    "seed": seed,
    "variant": variant,
    "accuracy": report["accuracy"],
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "latency_ms": float(np.mean(trial_means)),
    "latency_trials_ms": trial_means.tolist(),
}
print(f"Result for {key}: {result}")

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[key] = result

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Saved {key} to {RESULTS_FILE}")
