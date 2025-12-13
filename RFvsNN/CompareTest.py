import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

import matplotlib.pyplot as plt


# -----------------------------
# Pooled multiclass ROC (OvR)
# -----------------------------
def pooled_roc_curve(y_true_int, y_proba, n_classes):
    y_true_bin = label_binarize(y_true_int, classes=np.arange(n_classes))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    return fpr, tpr


def auroc_scores(y_true_int, y_proba, n_classes):
    y_true_bin = label_binarize(y_true_int, classes=np.arange(n_classes))
    auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    auc_weighted = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
    return float(auc_macro), float(auc_weighted)


# -----------------------------
# Load data
# -----------------------------
CSV_PATH = "spotify_songs.csv"
df = pd.read_csv(CSV_PATH)

print("Full dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

df["track_artist"] = df["track_artist"].fillna("Unknown_Artist")
df["playlist_genre"] = df["playlist_genre"].fillna("Unknown_Genre")

# -----------------------------
# FEATURE SETUP
# Use SAME FEATURES for both models (no raw text)
# -----------------------------
artist_le = LabelEncoder()
df["artist_encoded"] = artist_le.fit_transform(df["track_artist"])

FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "track_popularity",
    "artist_encoded",
]

TARGET = "playlist_genre"

needed_cols = FEATURES + [TARGET, "playlist_id"]
df_clean = df.dropna(subset=needed_cols).copy()

X = df_clean[FEATURES].copy()
y = df_clean[TARGET].copy()
groups = df_clean["playlist_id"].copy()

# Encode target
y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)
class_names = y_le.classes_
n_classes = len(class_names)

print("\nFeature matrix shape:", X.shape)
print("Classes:", class_names)

# -----------------------------
# Leakage-safe split (same for both)
# -----------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_enc[train_idx], y_enc[test_idx]

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# -----------------------------
# 1) Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)
rf_pred = np.argmax(rf_proba, axis=1)
rf_acc = accuracy_score(y_test, rf_pred)

rf_auc_macro, rf_auc_weighted = auroc_scores(y_test, rf_proba, n_classes)
rf_fpr, rf_tpr = pooled_roc_curve(y_test, rf_proba, n_classes)

print(f"\nRF Accuracy: {rf_acc:.4f}")
print(f"RF Macro AUROC: {rf_auc_macro:.4f} | RF Weighted AUROC: {rf_auc_weighted:.4f}")

# -----------------------------
# 2) Neural Net (MLP) — scaling required
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

nn = MLPClassifier(
    hidden_layer_sizes=(256, 256),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=0.0007,
    max_iter=600,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42,
    verbose=False,
)

nn.fit(X_train_s, y_train)
nn_proba = nn.predict_proba(X_test_s)
nn_pred = np.argmax(nn_proba, axis=1)
nn_acc = accuracy_score(y_test, nn_pred)

nn_auc_macro, nn_auc_weighted = auroc_scores(y_test, nn_proba, n_classes)
nn_fpr, nn_tpr = pooled_roc_curve(y_test, nn_proba, n_classes)

print(f"\nNN Accuracy: {nn_acc:.4f}")
print(f"NN Macro AUROC: {nn_auc_macro:.4f} | NN Weighted AUROC: {nn_auc_weighted:.4f}")

# -----------------------------
# Plot BOTH ROC curves on one graph
# -----------------------------
plt.figure(figsize=(7, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (Macro AUROC={rf_auc_macro:.3f})")
plt.plot(nn_fpr, nn_tpr, label=f"Neural Net (Macro AUROC={nn_auc_macro:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Genre Classification ROC (OvR pooled) — RF vs NN")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# SUBGENRE: RF vs NN ROC comparison (FLAT models)
# =========================================================

TARGET_SUB = "playlist_subgenre"

needed_cols = FEATURES + [TARGET_SUB, "playlist_id"]
df_sub = df.dropna(subset=needed_cols).copy()

X_sub = df_sub[FEATURES].copy()
y_sub = df_sub[TARGET_SUB].copy()
groups_sub = df_sub["playlist_id"].copy()

# Encode subgenre labels
sub_le = LabelEncoder()
y_sub_enc = sub_le.fit_transform(y_sub)
sub_classes = sub_le.classes_
n_sub_classes = len(sub_classes)

print("\nSUBGENRE Feature matrix shape:", X_sub.shape)
print("Number of subgenres:", n_sub_classes)
print("Example subgenres:", sub_classes[:10])

# -----------------------------
# Same leakage-safe split
# -----------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X_sub, y_sub_enc, groups=groups_sub))

X_sub_train = X_sub.iloc[train_idx]
X_sub_test = X_sub.iloc[test_idx]
y_sub_train = y_sub_enc[train_idx]
y_sub_test = y_sub_enc[test_idx]

print("Subgenre train size:", X_sub_train.shape, "test size:", X_sub_test.shape)

# =========================================================
# 1) Random Forest — SUBGENRE
# =========================================================
rf_sub = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

rf_sub.fit(X_sub_train, y_sub_train)
rf_sub_proba = rf_sub.predict_proba(X_sub_test)
rf_sub_pred = np.argmax(rf_sub_proba, axis=1)

rf_sub_acc = accuracy_score(y_sub_test, rf_sub_pred)
rf_sub_auc_macro, rf_sub_auc_weighted = auroc_scores(
    y_sub_test, rf_sub_proba, n_sub_classes
)
rf_sub_fpr, rf_sub_tpr = pooled_roc_curve(
    y_sub_test, rf_sub_proba, n_sub_classes
)

print(f"\nRF Subgenre Accuracy: {rf_sub_acc:.4f}")
print(
    f"RF Subgenre AUROC — Macro: {rf_sub_auc_macro:.4f}, "
    f"Weighted: {rf_sub_auc_weighted:.4f}"
)

# =========================================================
# 2) Neural Network — SUBGENRE (FLAT)
# =========================================================
scaler_sub = StandardScaler()
X_sub_train_s = scaler_sub.fit_transform(X_sub_train)
X_sub_test_s = scaler_sub.transform(X_sub_test)

nn_sub = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=600,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42,
    verbose=False,
)

nn_sub.fit(X_sub_train_s, y_sub_train)
nn_sub_proba = nn_sub.predict_proba(X_sub_test_s)
nn_sub_pred = np.argmax(nn_sub_proba, axis=1)

nn_sub_acc = accuracy_score(y_sub_test, nn_sub_pred)
nn_sub_auc_macro, nn_sub_auc_weighted = auroc_scores(
    y_sub_test, nn_sub_proba, n_sub_classes
)
nn_sub_fpr, nn_sub_tpr = pooled_roc_curve(
    y_sub_test, nn_sub_proba, n_sub_classes
)

print(f"\nNN Subgenre Accuracy: {nn_sub_acc:.4f}")
print(
    f"NN Subgenre AUROC — Macro: {nn_sub_auc_macro:.4f}, "
    f"Weighted: {nn_sub_auc_weighted:.4f}"
)

# =========================================================
# Plot RF vs NN ROC — SUBGENRE
# =========================================================
plt.figure(figsize=(7, 6))
plt.plot(
    rf_sub_fpr,
    rf_sub_tpr,
    label=f"Random Forest (Macro AUROC={rf_sub_auc_macro:.3f})",
)
plt.plot(
    nn_sub_fpr,
    nn_sub_tpr,
    label=f"Neural Net (Macro AUROC={nn_sub_auc_macro:.3f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Subgenre Classification ROC (OvR pooled) — RF vs NN")
plt.legend()
plt.tight_layout()
plt.show()
