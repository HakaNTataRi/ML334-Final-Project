import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt

# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv("spotify_songs.csv")

print(df.shape)
print(df.head())

# =========================================================
# 2. Select features + targets
# =========================================================

# All numeric features we WILL use as inputs
feature_cols = [
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
    "track_popularity"
]

# Just to be safe: intersect with existing columns
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y_genre = df["playlist_genre"].copy()
y_sub = df["playlist_subgenre"].copy()

print("Feature matrix shape:", X.shape)
print("Example genres:", y_genre.unique()[:10])
print("Example subgenres:", y_sub.unique()[:10])

# =========================================================
# 3. Encode labels
# =========================================================

# Encode genre labels
genre_encoder = LabelEncoder()
y_genre_enc = genre_encoder.fit_transform(y_genre)

# Encode subgenre labels
sub_encoder = LabelEncoder()
y_sub_enc = sub_encoder.fit_transform(y_sub)

# =========================================================
# 5A. GENRE: 5-Fold Cross Validation to Measure True Accuracy
# =========================================================
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("\nRunning 5-fold cross-validation for genre classification...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_genre_enc), 1):

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y_genre_enc[train_idx], y_genre_enc[test_idx]

    # Scale inside each fold (VERY important to avoid leakage)
    scaler_cv = StandardScaler()
    X_tr_s = scaler_cv.fit_transform(X_tr)
    X_te_s = scaler_cv.transform(X_te)

    model_cv = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=400,
        random_state=42,
        verbose=False
    )

    model_cv.fit(X_tr_s, y_tr)
    preds = model_cv.predict(X_te_s)

    acc = accuracy_score(y_te, preds)
    cv_scores.append(acc)

    print(f"Fold {fold} accuracy: {acc:.4f}")

print("\nCross-validation accuracy scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("Std deviation:", np.std(cv_scores))

# =========================================================
# 4. Train/test split
# =========================================================
X_train, X_test, y_genre_train, y_genre_test, y_sub_train, y_sub_test = train_test_split(
    X,
    y_genre_enc,
    y_sub_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_genre_enc  # stratify by genre to keep class balance
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# =========================================================
# 4.5 Class weights -> sample weights (handle imbalance)
# =========================================================

# Genre sample weights
classes_genre = np.unique(y_genre_train)
class_weights_genre = compute_class_weight(
    class_weight="balanced",
    classes=classes_genre,
    y=y_genre_train
)
genre_weight_dict = {c: w for c, w in zip(classes_genre, class_weights_genre)}
genre_sample_weight = np.array([genre_weight_dict[y] for y in y_genre_train])

# Subgenre sample weights
classes_sub = np.unique(y_sub_train)
class_weights_sub = compute_class_weight(
    class_weight="balanced",
    classes=classes_sub,
    y=y_sub_train
)
sub_weight_dict = {c: w for c, w in zip(classes_sub, class_weights_sub)}
sub_sample_weight = np.array([sub_weight_dict[y] for y in y_sub_train])

# =========================================================
# 5. Scale features
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 6. Neural network for playlist_genre (tuned)
# =========================================================
nn_genre = MLPClassifier(
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
    verbose=True
)

nn_genre.fit(X_train_scaled, y_genre_train)  # or with sample_weight if you kept that


# Predictions
y_genre_pred = nn_genre.predict(X_test_scaled)

# Evaluation
genre_acc = accuracy_score(y_genre_test, y_genre_pred)
print(f"\nGenre NN Accuracy (tuned): {genre_acc:.4f}")

print("\nClassification report (genre, tuned):")
print(classification_report(y_genre_test, y_genre_pred, target_names=genre_encoder.classes_))

# =========================================================
# 7.1 Train one subgenre model per GENRE (hierarchical)
# =========================================================

# unique genre ids in the training set
classes_genre = np.unique(y_genre_train)

sub_models = {}            # maps genre_id -> trained MLP for subgenres
genre_to_subclasses = {}   # maps genre_id -> list of subgenre ids used by that model

for g in classes_genre:
    # mask for this genre in the training set
    mask = (y_genre_train == g)
    X_g = X_train_scaled[mask]
    y_sub_g = y_sub_train[mask]

    # which subgenre labels appear for this genre?
    sub_classes_g = np.unique(y_sub_g)
    genre_to_subclasses[g] = sub_classes_g

    print(
        f"\nTraining subgenre model for genre id {g} "
        f"({genre_encoder.inverse_transform([g])[0]}) "
        f"with {len(y_sub_g)} samples and {len(sub_classes_g)} subgenres"
    )

    # compute class weights within this genre subset
    class_weights_sub_g = compute_class_weight(
        class_weight="balanced",
        classes=sub_classes_g,
        y=y_sub_g
    )
    weight_map_g = {c: w for c, w in zip(sub_classes_g, class_weights_sub_g)}
    sample_weight_g = np.array([weight_map_g[y] for y in y_sub_g])

    # subgenre model for this genre
    model_g = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )

    model_g.fit(X_g, y_sub_g, sample_weight=sample_weight_g)

    # store the trained model
    sub_models[g] = model_g

print("\nFinished training per-genre subgenre models.")

# =========================================================
# 7.2 Hierarchical prediction: genre -> subgenre
# =========================================================

hier_sub_preds = []

for i in range(len(X_test_scaled)):
    x = X_test_scaled[i].reshape(1, -1)

    # 1) predict main genre
    g_pred = nn_genre.predict(x)[0]

    # 2) get the subgenre model for that predicted genre
    model_g = sub_models[g_pred]

    # 3) predict subgenre with that genre-specific model
    sub_pred = model_g.predict(x)[0]
    hier_sub_preds.append(sub_pred)

hier_sub_preds = np.array(hier_sub_preds)

# Evaluate hierarchical subgenre performance
from sklearn.metrics import accuracy_score, classification_report

hier_sub_acc = accuracy_score(y_sub_test, hier_sub_preds)
print(f"\nSubgenre Hierarchical NN Accuracy: {hier_sub_acc:.4f}")

print("\nClassification report (subgenre, hierarchical):")
print(classification_report(
    y_sub_test,
    hier_sub_preds,
    target_names=sub_encoder.classes_
))

# =========================================================
# 8. Confusion matrix plotting (no seaborn)
# =========================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    # Force confusion_matrix to include ALL classes in the given order
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Tick marks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Text in each cell
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if val > max_val / 2 else "black",
                fontsize=8
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# Plot for genre
plot_confusion_matrix(
    y_true=y_genre_test,
    y_pred=y_genre_pred,
    class_names=genre_encoder.classes_,
    title="Confusion Matrix – Genre (Neural Network)"
)

# Plot for subgenre
plot_confusion_matrix(
    y_true=y_sub_test,
    y_pred=hier_sub_preds,
    class_names=sub_encoder.classes_,
    title="Confusion Matrix – Subgenre (Hierarchical NN)"
)
