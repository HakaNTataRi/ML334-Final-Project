import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# 0. Load data
# ============================

CSV_PATH = "spotify_songs.csv"  # change if needed
df = pd.read_csv(CSV_PATH)

# Basic cleaning for text fields
df["track_artist"] = df["track_artist"].fillna("Unknown_Artist")
df["playlist_name"] = df["playlist_name"].fillna("Unknown_Playlist")
df["playlist_subgenre"] = df["playlist_subgenre"].fillna("Unknown_Subgenre")

# ============================
# 1. Extra categorical encodings
# ============================

artist_le = LabelEncoder()
df["artist_encoded"] = artist_le.fit_transform(df["track_artist"])

playlist_le = LabelEncoder()
df["playlist_encoded"] = playlist_le.fit_transform(df["playlist_name"])

# Keep this in case you want to predict subgenre as a target,
# but DO NOT use subgenre_encoded as a feature for predicting genre.
subgenre_le = LabelEncoder()
df["subgenre_encoded"] = subgenre_le.fit_transform(df["playlist_subgenre"])

# Numeric audio/track features
NUMERIC_FEATURES = [
    "track_popularity",
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
]

# Encoded categorical features (safe to use as inputs)
CATEGORICAL_ENC_FEATURES = [
    "artist_encoded",
    "playlist_encoded",
]

BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_ENC_FEATURES


# ============================
# Helper: Random Forest (with PCA) for a target
# ============================

def run_rf_for_target(df, target_col, title_prefix):
    print("\n" + "=" * 80)
    print(f"{title_prefix}: Target = {target_col}")
    print("=" * 80)

    # Drop rows missing target
    df_clean = df.dropna(subset=[target_col]).copy()

    # Choose features (NO subgenre_encoded to avoid leakage)
    features = BASE_FEATURES.copy()

    X = df_clean[features]
    y = df_clean[target_col]

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Grouped split by playlist_id to avoid playlist-level leakage
    groups = df_clean["playlist_id"]  # group key

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y_enc[train_idx]
    y_test = y_enc[test_idx]

    # ----------------------------
    # 1) Baseline Random Forest
    # ----------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print("\n=== BASELINE RANDOM FOREST ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))

    # Confusion Matrix - RF
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"{title_prefix} - Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Per-class TP/FP/FN/TN — only for GENRE (not subgenre)
    if target_col == "playlist_genre":
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        metrics_df = pd.DataFrame({
            "Genre": le.classes_,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        })

        print("\nPer-genre confusion components (Random Forest):")
        print(metrics_df)

    # ----------------------------
    # 2) PCA + Random Forest
    # ----------------------------
    pipe = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=42)),  # keep 95% variance
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipe.fit(X_train, y_train)
    pca_pred = pipe.predict(X_test)

    print("\n=== PCA + RANDOM FOREST ===")
    print("Accuracy:", accuracy_score(y_test, pca_pred))
    print(classification_report(y_test, pca_pred, target_names=le.classes_))

    # Confusion Matrix - PCA + RF
    cm_pca = confusion_matrix(y_test, pca_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_pca,
        annot=False,
        cmap="Oranges",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"{title_prefix} - PCA + Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ============================
# Helper: RF with text (playlist_name TF-IDF)
# ============================

def run_rf_with_text(df, target_col, title_prefix):
    print("\n" + "=" * 80)
    print(f"{title_prefix} WITH TEXT: Target = {target_col}")
    print("=" * 80)

    df_clean = df.dropna(subset=[target_col]).copy()
    y = df_clean[target_col]

    numeric_cols = NUMERIC_FEATURES  # audio + popularity
    text_col = "playlist_name"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col),
        ],
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        df_clean[[text_col] + numeric_cols],
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("rf", rf),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("Accuracy (RF + playlist_name TF-IDF):", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))


# ============================
# Helper: Hyperparameter tuning for RF
# ============================

def tune_rf(df, target_col):
    print("\nTUNING RANDOM FOREST for", target_col)

    df_clean = df.dropna(subset=[target_col]).copy()
    features = BASE_FEATURES.copy()

    X = df_clean[features]
    y = df_clean[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 10, 20, 40],
        "max_features": ["sqrt", "log2", 0.5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    best_rf = search.best_estimator_

    pred = best_rf.predict(X_test)
    print("Tuned RF accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))


# ============================
# Helper: kNN (with scaling + PCA)
# ============================

def run_knn(df, target_col, title_prefix):
    print("\n" + "=" * 80)
    print(f"{title_prefix}: kNN Target = {target_col}")
    print("=" * 80)

    df_clean = df.dropna(subset=[target_col]).copy()

    features = BASE_FEATURES.copy()
    # DO NOT append subgenre_encoded here — avoid leakage
    X = df_clean[features]
    y = df_clean[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),  # optional
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ])

    knn_pipe.fit(X_train, y_train)
    pred = knn_pipe.predict(X_test)

    print("kNN Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))


# ============================
# Helper: Neural Network (MLP)
# ============================

def run_nn(df, target_col, title_prefix):
    print("\n" + "=" * 80)
    print(f"{title_prefix}: NN Target = {target_col}")
    print("=" * 80)

    df_clean = df.dropna(subset=[target_col]).copy()

    features = BASE_FEATURES.copy()
    # DO NOT append subgenre_encoded here — avoid leakage
    X = df_clean[features]
    y = df_clean[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    nn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=50,
            random_state=42,
        )),
    ])

    nn_pipe.fit(X_train, y_train)
    pred = nn_pipe.predict(X_test)

    print("NN Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))


# ============================
# Main: run experiments
# ============================

if __name__ == "__main__":
    # Random Forest + PCA: GENRE
    run_rf_for_target(df, target_col="playlist_genre", title_prefix="GENRE PREDICTION")

    # Random Forest + PCA: SUBGENRE
    run_rf_for_target(df, target_col="playlist_subgenre", title_prefix="SUBGENRE PREDICTION")

    # RF + playlist_name TF-IDF (genre only)
    # Uncomment if you want to run this:
    # run_rf_with_text(df, target_col="playlist_genre", title_prefix="GENRE PREDICTION")

    # Hyperparameter tuning for RF (genre)
    # Warning: this can be slow; uncomment when ready.
    # tune_rf(df, target_col="playlist_genre")

    # kNN with scaling + PCA (genre)
    # run_knn(df, target_col="playlist_genre", title_prefix="GENRE PREDICTION")

    # Neural network (MLP) (genre)
    # run_nn(df, target_col="playlist_genre", title_prefix="GENRE PREDICTION")
