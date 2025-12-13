import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# =========================================================
# Confusion matrix plotting 
# =========================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > max_val / 2 else "black",
                fontsize=8
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return cm


# =========================================================
# ROC/AUROC for multiclass OvR + pooled ROC curve
# =========================================================
def report_and_plot_auroc(y_true_int, y_proba, class_names, title="ROC (OvR)"):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_int, classes=np.arange(n_classes))

    auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    auc_weighted = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
    print(f"{title} | Macro AUROC: {auc_macro:.4f} | Weighted AUROC: {auc_weighted:.4f}")

    # Pooled ROC (flatten OvR decisions)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{title} (Macro AUROC≈{auc_macro:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc_macro, auc_weighted


# =========================================================
# 0. Load data + prints
# =========================================================
CSV_PATH = "spotify_songs.csv"
df = pd.read_csv(CSV_PATH)

print("Full dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

df["track_artist"] = df["track_artist"].fillna("Unknown_Artist")
df["playlist_genre"] = df["playlist_genre"].fillna("Unknown_Genre")
df["playlist_subgenre"] = df["playlist_subgenre"].fillna("Unknown_Subgenre")

print("\nFeature matrix will be built from audio + artist.")
print("\nExample genres:", df["playlist_genre"].unique()[:10])
print("Example subgenres:", df["playlist_subgenre"].unique()[:10])


# =========================================================
# 1. Features (shared setup)
# Keep track_artist via artist_encoded (NOT raw text)
# =========================================================
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


# =========================================================
# 2. 5-Fold CV (Stratified) for GENRE — same style as NN output
# NOTE: this is for model-to-model comparison; final eval still uses Group split.
# =========================================================
def rf_5fold_cv_genre(df):
    df_cv = df.dropna(subset=FEATURES + ["playlist_genre"]).copy()

    X = df_cv[FEATURES].copy()
    y = df_cv["playlist_genre"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("\nFeature matrix shape:", X.shape)
    print("Example genres:", le.classes_[:10])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    print("\nRunning 5-fold cross-validation for Random Forest genre classification...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        rf_cv = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        rf_cv.fit(X_tr, y_tr)
        preds = rf_cv.predict(X_te)

        acc = accuracy_score(y_te, preds)
        cv_scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    print("\nCV scores:", cv_scores)
    print("Mean CV Accuracy:", float(np.mean(cv_scores)))
    print("Std deviation:", float(np.std(cv_scores)))


# =========================================================
# 3. Leakage-safe evaluation: GroupShuffleSplit by playlist_id
# =========================================================
def run_rf_for_target(df, target_col, title_prefix, do_pca=True):
    print("\n" + "=" * 80)
    print(f"{title_prefix}: Target = {target_col}")
    print("=" * 80)

    needed_cols = FEATURES + [target_col, "playlist_id"]
    df_clean = df.dropna(subset=needed_cols).copy()

    X = df_clean[FEATURES].copy()
    y = df_clean[target_col].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    groups = df_clean["playlist_id"]
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y_enc[train_idx]
    y_test = y_enc[test_idx]

    print("Train size:", X_train.shape, "Test size:", X_test.shape)
    print("Classes:", le.classes_[:10])

    # ----------------------------
    # Baseline RF (NO PCA)
    # ----------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)

    print("\n=== BASELINE RANDOM FOREST (no PCA) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))

    cm = plot_confusion_matrix(
        y_true=y_test,
        y_pred=pred,
        class_names=le.classes_,
        title=f"{title_prefix} – Confusion Matrix (Baseline RF)"
    )

    # Per-class TP/FP/FN/TN only for GENRE
    if target_col == "playlist_genre":
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        metrics_df = pd.DataFrame({"Genre": le.classes_, "TP": TP, "FP": FP, "FN": FN, "TN": TN})
        print("\nPer-genre confusion components (Baseline RF):")
        print(metrics_df)

    report_and_plot_auroc(
        y_true_int=y_test,
        y_proba=proba,
        class_names=le.classes_,
        title=f"{title_prefix} – Baseline RF ROC (OvR)"
    )

    # ----------------------------
    # PCA + RF (comparison)
    # ----------------------------
    if do_pca:
        pipe = Pipeline([
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])

        pipe.fit(X_train, y_train)
        pca_pred = pipe.predict(X_test)
        pca_proba = pipe.predict_proba(X_test)

        n_comps = pipe.named_steps["pca"].n_components_
        explained = float(pipe.named_steps["pca"].explained_variance_ratio_.sum())
        print(f"\nPCA kept {n_comps} components (explained variance={explained:.4f}).")

        print("\n=== PCA + RANDOM FOREST ===")
        print("Accuracy:", accuracy_score(y_test, pca_pred))
        print(classification_report(y_test, pca_pred, target_names=le.classes_))

        plot_confusion_matrix(
            y_true=y_test,
            y_pred=pca_pred,
            class_names=le.classes_,
            title=f"{title_prefix} – Confusion Matrix (PCA + RF)"
        )

        report_and_plot_auroc(
            y_true_int=y_test,
            y_proba=pca_proba,
            class_names=le.classes_,
            title=f"{title_prefix} – PCA + RF ROC (OvR)"
        )


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # 5-fold CV (genre only) to match NN-style output
    rf_5fold_cv_genre(df)

    # Leakage-safe evaluations + plots
    run_rf_for_target(df, "playlist_genre", "GENRE PREDICTION", do_pca=True)
    run_rf_for_target(df, "playlist_subgenre", "SUBGENRE PREDICTION", do_pca=True)

