from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

import shap
import matplotlib.pyplot as plt

from load_data import load_all
from feature_engineering import build_features
from merge_all import merge_with_user_repo


# ---------
# Chemins
# ---------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# --------------------------
# Recharger data + features
# --------------------------
dfs = load_all(DATA_ROOT)

# Features PR (structure, commits, collaboration, issues, temps)
pr_feats = build_features(dfs, agent_filter=None)

# Merge avec user + repo
df = merge_with_user_repo(pr_feats, dfs["user"], dfs["repo"])

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
    "forks", "stars",
]

feature_cols = [c for c in candidate_features if c in df.columns]
target_col = "accepted_pr"

df_model = df[feature_cols + [target_col]].copy()
df_model = df_model.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_model[feature_cols].astype(float)
y = df_model[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() == 2 else None,
)

print(f"Dataset d'évaluation : {X.shape[0]} lignes, {X.shape[1]} features")


# -------------------
# Charger le modèle
# -------------------
model_path = ARTIFACTS / "model_rf.joblib"
clf = joblib.load(model_path)
print(f"Modèle chargé depuis {model_path}")


# ---------------------
# 3) Évaluation simple
# ---------------------
y_pred = clf.predict(X_test)
print("\n=== Classification report (évaluation) ===")
report = classification_report(y_test, y_pred, digits=3)
print(report)

# Sauvegarde du rapport dans un fichier texte
with open(ARTIFACTS / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)


# ------------------------------
# SHAP : importance globale
# ------------------------------
shap.initjs()

sample_size = min(2000, X_train.shape[0])
X_train_sample = X_train.sample(sample_size, random_state=42)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train_sample)

print("\n[SHAP] type(shap_values) :", type(shap_values))

if isinstance(shap_values, list):
    shap_for_plot = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    print("[SHAP] shap_values shape :", shap_values.shape)
    shap_for_plot = shap_values[:, :, 1]
else:
    shap_for_plot = shap_values

print("[SHAP] X_train_sample shape :", X_train_sample.shape)
print("[SHAP] shap_for_plot shape  :", shap_for_plot.shape)

# Summary plot bar: Importance moyenne absolue
plt.figure()
shap.summary_plot(
    shap_for_plot,
    X_train_sample,
    feature_names=X_train_sample.columns,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(ARTIFACTS / "shap_summary_bar.png", bbox_inches="tight")
plt.close()

print("Graphique SHAP (bar) sauvegardé dans artifacts/shap_summary_bar.png")


# ------------------------------
# Gradients : Importance par permutation
# ------------------------------
print("\n=== Importance par permutation (sensibilité globale) ===")
perm = permutation_importance(
    clf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

perm_importances = pd.Series(
    perm.importances_mean,
    index=feature_cols
).sort_values(ascending=False)

print(perm_importances)

perm_importances.to_csv(ARTIFACTS / "permutation_importances.csv")
print("Importances par permutation sauvegardées dans artifacts/permutation_importances.csv")
