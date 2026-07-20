import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/saved_artifacts/MCDNN_pruned_seed42.tflite", help="Target path to compiled TFLite artifact")
args = parser.parse_args()

# Fixed data path alignment
x_test = np.load("data/x_test_real.npy").astype(np.float32)
y_test = np.load("data/y_test_real.npy")

n_classes = y_test.shape[1]
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

y_score = []
print(f"Streaming execution path for: {args.model}")
for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], [x_test[i]])
    interpreter.invoke()
    y_score.append(interpreter.get_tensor(output_details[-1]['index'])[0])
y_score = np.array(y_score)

true_labels = np.argmax(y_test, axis=1)
pred_labels = np.argmax(y_score, axis=1)

# Generate Heatmap Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f"23-Class Confusion Matrix\n({args.model.split('/')[-1]})")
plt.ylabel('Actual Attack Paradigm Class')
plt.xlabel('Predicted Attack Paradigm Class')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to root directory.")
