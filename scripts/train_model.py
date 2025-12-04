from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from load_data import load_all
from feature_engineering import build_features
from merge_all import merge_with_user_repo


# -------------
# REPERTOIRES
# -------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
ARTIFACTS = BASE_DIR / "artifacts"


# -----------------------
# CHARGEMENT DES DONNÉES
# -----------------------
dfs = load_all(DATA_ROOT)


# -----------------------
# FEATURE ENGINEERING
# -----------------------
pr_feats = build_features(dfs, agent_filter=None)


# ------------------
# MERGE USER + REPO
# ------------------
df = merge_with_user_repo(pr_feats, dfs["user"], dfs["repo"])


# ------------------------
# SELECTION DES FEATURES
# ------------------------

candidate_features = [

    # Structure PR
    "title_length", "body_length",

    # Commits / fichiers / patchs
    "commits", "changed_files", "additions", "deletions", "total_changes",

    # Collaboration
    "num_comments", "num_review_comments",
    "num_reviews", "num_reviewers_unique",

    # Issues
    "has_issue_linked",

    # Temporalité
    "pr_duration_days", "created_hour",

    # Auteur
    "followers", "public_repos", "author_tenure_days",

    # Repo
    "forks", "stars"
]

# Ajout des colonnes pour les agents IA
agent_cols = [c for c in df.columns if c.startswith("agent_")]
candidate_features += agent_cols

feature_cols = [c for c in candidate_features if c in df.columns]

target_col = "accepted_pr"

df_model = df[feature_cols + [target_col]].copy()


# -----------
# NETTOYAGE
# -----------
df_model = df_model.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_model[feature_cols].astype(float)
y = df_model[target_col].astype(int)

print(f"\nDataset final : {X.shape[0]} lignes, {X.shape[1]} features")


# ----------------------
# TRAIN / TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() == 2 else None
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

clf.fit(X_train, y_train)


# ------------
# EVALUATION
# ------------
y_pred = clf.predict(X_test)
print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, digits=3))


# --------------------
# FEATURE IMPORTANCES
# --------------------
imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Top feature importances ===")
print(imp.head(25))


# -----------
# SAUVEGARDE
# -----------
ARTIFACTS.mkdir(parents=True, exist_ok=True)

joblib.dump(clf, ARTIFACTS / "model_rf.joblib")
pd.Series(feature_cols).to_csv(ARTIFACTS / "model_features.csv", index=False)

print(f"\n Modèle sauvegardé dans {ARTIFACTS/'model_rf.joblib'}")
