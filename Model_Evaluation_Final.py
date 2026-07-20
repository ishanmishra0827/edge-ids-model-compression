import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

x_test = np.load("x_test_real.npy").astype(np.float32)
y_test = np.load("y_test_real.npy")

interpreter = tf.lite.Interpreter(model_path="MCDNN_final.tflite")
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
print("Running TFLite inference...")

for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], [x_test[i]])
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(head3_index)
    y_pred.append(np.argmax(output_data))

y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('MCDNN Multi-Class Confusion Matrix (74.53% Sparsity Exit Node)')
plt.ylabel('Actual Attack Paradigm Class')
plt.xlabel('Predicted Attack Paradigm Class')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
