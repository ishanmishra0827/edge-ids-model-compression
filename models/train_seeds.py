import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
import time
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras
from tf_keras import layers
from sklearn.metrics import classification_report

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_FILE = "benchmarks/multiseed_results.json"

# Ensure output paths exist safely
os.makedirs("models/saved_artifacts", exist_ok=True)
os.makedirs("benchmarks", exist_ok=True)

# Data Ingestion
x_train = np.load("data/x_train.npy").astype(np.float32)
y_train = np.load("data/y_train.npy").astype(np.float32)
x_val = np.load("data/x_val.npy").astype(np.float32)
y_val = np.load("data/y_val.npy").astype(np.float32)
x_test_real = np.load("data/x_test_real.npy").astype(np.float32)
y_test_real = np.load("data/y_test_real.npy")
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
            interpreter.set_tensor(input_index, sample.reshape(1, -1))
            interpreter.invoke()
        end_time = time.perf_counter()
        trial_means.append(((end_time - start_time) * 1000) / len(test_samples))
    return float(np.mean(trial_means)), np.array(trial_means).tolist()

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
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"], "weighted_f1": report["weighted avg"]["f1-score"]}

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
        
    model.compile(optimizer='adam', loss=['categorical_crossentropy']*3, metrics=['accuracy'], loss_weights=[0.2, 0.3, 0.5])
    model.fit(x_train, [y_train]*3, epochs=15, batch_size=128, validation_data=(x_val, [y_val]*3), callbacks=fit_callbacks, verbose=0)
    
    if use_pruning:
        model = tfmot.sparsity.keras.strip_pruning(model)
        
    # Save the base model weights
    model.save(f"models/saved_artifacts/MCDNN_{key}.h5")
    
    tflite_path = f"models/saved_artifacts/MCDNN_{key}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    metrics = evaluate_tflite(tflite_path)
    latency_mean, trial_means = profile_tflite_latency(tflite_path)
    
    return key, {"seed": seed, "variant": variant, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"], "latency_ms": latency_mean, "latency_trials_ms": trial_means}

results = load_results()
for seed in SEEDS:
    for use_pruning in [False, True]:
        key = f"{'pruned' if use_pruning else 'dense'}_seed{seed}"
        if key in results:
            print(f"Skipping {key} (completed)")
            continue
        key, result = run_one(seed, use_pruning)
        results[key] = result
        save_results(results)
print("\nALL MULTI-SEED RUNS COMPLETE.")
