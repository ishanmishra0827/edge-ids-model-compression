import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
test_url = "https://raw.githubusercontent.com/arjbah/nsl-kdd/master/nsl-kdd/KDDTest+.txt"

print("Downloading KDDTrain+...")
df_train = pd.read_csv(train_url, header=None)
df_train.drop(df_train.columns[-1], axis=1, inplace=True)

print("Downloading KDDTest+...")
df_test = pd.read_csv(test_url, header=None)
df_test.drop(df_test.columns[-1], axis=1, inplace=True) 

print(f"Train records: {len(df_train)}  |  Test records: {len(df_test)}")

categorical_cols = [1, 2, 3]
for col in categorical_cols:
    df_train[col] = df_train[col].astype(str)
    df_test[col] = df_test[col].astype(str)

combined = pd.concat([df_train, df_test], keys=["train", "test"])
combined_encoded = pd.get_dummies(combined, columns=categorical_cols)

df_train_encoded = combined_encoded.xs("train")
df_test_encoded = combined_encoded.xs("test")

x_train_full = df_train_encoded.drop(41, axis=1).values
x_test_full = df_test_encoded.drop(41, axis=1).values

y_train_labels = df_train_encoded[41].values
y_test_labels = df_test_encoded[41].values

train_classes = sorted(pd.unique(y_train_labels))
total_classes = len(train_classes)
label_to_idx = {label: i for i, label in enumerate(train_classes)}

print(f"\nTotal classes learned from training set: {total_classes}")
print(f"Training classes: {train_classes}")

known_mask = pd.Series(y_test_labels).isin(label_to_idx).values
unseen_mask = ~known_mask

n_known = known_mask.sum()
n_unseen = unseen_mask.sum()
print(f"\nTest records with a KNOWN (trainable) label: {n_known}")
print(f"Test records with an UNSEEN attack type:       {n_unseen}")

if n_unseen > 0:
    unseen_labels, unseen_counts = np.unique(y_test_labels[unseen_mask], return_counts=True)
    unseen_report = pd.DataFrame({"attack_type": unseen_labels, "count": unseen_counts})
    unseen_report = unseen_report.sort_values("count", ascending=False)
    unseen_report.to_csv("unseen_class_report.csv", index=False)
    print("\nUnseen attack types in test set (never appeared in training):")
    print(unseen_report.to_string(index=False))
    print("\n-> Saved as 'unseen_class_report.csv'.")

x_test_real = x_test_full[known_mask]
y_test_ints_real = np.array([label_to_idx[label] for label in y_test_labels[known_mask]])
y_test_real = np.eye(total_classes)[y_test_ints_real]

print(f"\nFinal x_test_real shape: {x_test_real.shape}")
print(f"Final y_test_real shape: {y_test_real.shape}")

from sklearn.model_selection import train_test_split

y_train_ints_all = np.array([label_to_idx[label] for label in y_train_labels])
y_train_onehot_all = np.eye(total_classes)[y_train_ints_all]

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train_full, y_train_onehot_all, test_size=0.15,
    stratify=y_train_ints_all, random_state=42
)

print("\nSynthesizing minority class instances via SMOTE... Please wait.")

y_tr_ints = np.argmax(y_tr, axis=1)
class_counts = pd.Series(y_tr_ints).value_counts()
min_class_size = class_counts.min()
safe_k = max(1, min(5, min_class_size - 1))
print(f"Smallest training class has {min_class_size} samples -> using k_neighbors={safe_k}")

smote = SMOTE(k_neighbors=safe_k, random_state=42)
x_train_res, y_train_res_ints = smote.fit_resample(x_tr, y_tr_ints)
y_train_res = np.eye(total_classes)[y_train_res_ints]

unique, counts = np.unique(y_train_res_ints, return_counts=True)
plt.figure(figsize=(12, 6))
plt.bar(unique, counts, color='steelblue', edgecolor='black')
plt.xlabel("Attack Class ID")
plt.ylabel("Sample Count")
plt.title("Balanced Class Distribution Post-SMOTE Optimization")
plt.tight_layout()
plt.savefig("balanced_classes.png")

np.save("x_train.npy", x_train_res.astype(np.float32))
np.save("y_train.npy", y_train_res.astype(np.float32))
np.save("x_val.npy", x_val.astype(np.float32))
np.save("y_val.npy", y_val.astype(np.float32))
np.save("x_test_real.npy", x_test_real.astype(np.float32))
np.save("y_test_real.npy", y_test_real.astype(np.float32))

print(f"\nPreprocessing complete.")
print(f"  Train (SMOTE-balanced): {x_train_res.shape}")
print(f"  Validation (held out from train, no SMOTE): {x_val.shape}")
print(f"  Test (REAL KDDTest+, known-class subset):    {x_test_real.shape}")
print(f"  Classes: {total_classes}")
