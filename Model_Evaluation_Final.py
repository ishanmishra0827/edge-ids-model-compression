import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "MCDNN_pruned_seed42.tflite"
    print(f"No model path given, defaulting to: {model_path}")
    print("Usage: python Model_Evaluation_Final.py <path_to_tflite_model>")

x_test = np.load("x_test_real.npy").astype(np.float32)
y_test = np.load("y_test_real.npy")

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
except ValueError:
    print(f"\nError: Could not find or load '{model_path}'.")
    print("Check that this file exists in the current directory, or pass")
    print("the correct path as a command-line argument.")
    sys.exit(1)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

head3_index = None
for detail in output_details:
    if 'head3' in detail['name']:
        head3_index = detail['index']
        break

if head3_index is None:
    head3_index = output_details[-1]['index']

y_pred = []
print(f"Running TFLite inference ({model_path})...")

for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], [x_test[i]])
    interpreter.invoke()

    output_data = interpreter.get_tensor(head3_index)
    y_pred.append(np.argmax(output_data))

y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'MCDNN Confusion Matrix — {model_path}')
plt.ylabel('Actual Attack Taxonomy Class')
plt.xlabel('Predicted Attack Taxonomy Class')
plt.savefig('labeled_confusion_matrix.png', dpi=300, bbox_inches='tight')

print("\nClassification report:")
print(classification_report(y_true, y_pred, zero_division=0))
