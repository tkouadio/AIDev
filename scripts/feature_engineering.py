import pandas as pd
import numpy as np

# ============
# UTILITAIRES
# ============

def _key_series(df: pd.DataFrame, candidates=("number", "pr_number", "id")) -> pd.Series:
   
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError("Aucune colonne clé PR trouvée dans PR DataFrame.")


def _extract_pr_number_from_url(df, url_col="pull_request_url"):

    if url_col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    return df[url_col].str.extract(r'/pulls/(\d+)$')[0].astype(float)


# =============================
# FEATURE ENGINEERING PRINCIPAL
# =============================

def build_features(dfs: dict, agent_filter: str | None = None) -> pd.DataFrame:
    """Calcule toutes les métriques nécessaires pour le modèle."""
    
    pr = dfs["pr"].copy()

    # ------------------------------
    # ENCODAGE DES AGENTS (One-Hot)
    # ------------------------------
    if "agent" in pr.columns:
        agent_dummies = pd.get_dummies(pr["agent"], prefix="agent")
        pr = pd.concat([pr, agent_dummies], axis=1)
    else:
        agent_dummies = pd.DataFrame(index=pr.index) 

    if agent_filter:
        if "agent" in pr.columns:
            pr = pr[pr["agent"] == agent_filter].copy()

    # -------
    # LABEL
    # -------
    pr["accepted_pr"] = ((pr["state"] == "closed") & pr["merged_at"].notna()).astype(int)

    # --------------
    # STRUCTURE PR
    # --------------
    pr["title_length"] = pr["title"].fillna("").str.len()
    pr["body_length"] = pr["body"].fillna("").str.len()

    # ========================================
    # COMMITS (via pr_commit_details.parquet)
    # ========================================
    commit_df = dfs["pr_commit_details"].copy()

    if not commit_df.empty:
        # nombre de commits uniques
        commits_map = commit_df.groupby("pr_id")["sha"].nunique()

        # fichiers modifiés
        changed_files_map = commit_df.groupby("pr_id")["filename"].nunique()

        # additions / deletions / changes
        adds_map = commit_df.groupby("pr_id")["additions"].sum()
        dels_map = commit_df.groupby("pr_id")["deletions"].sum()
        changes_map = commit_df.groupby("pr_id")["changes"].sum()

        pr["commits"] = pr["id"].map(commits_map).fillna(0)
        pr["changed_files"] = pr["id"].map(changed_files_map).fillna(0)
        pr["additions"] = pr["id"].map(adds_map).fillna(0)
        pr["deletions"] = pr["id"].map(dels_map).fillna(0)
        pr["total_changes"] = pr["id"].map(changes_map).fillna(0)
    else:
        pr["commits"] = 0
        pr["changed_files"] = 0
        pr["additions"] = 0
        pr["deletions"] = 0
        pr["total_changes"] = 0

    # =================
    # 4) COLLABORATION
    # =================
   
    # Commentaires généraux
    comments_df = dfs["pr_comments"].copy()
    if "pr_id" in comments_df.columns:
        comments_map = comments_df.groupby("pr_id").size()
        pr["num_comments"] = pr["id"].map(comments_map).fillna(0).astype(int)
    else:
        pr["num_comments"] = 0

    # Reviews
    reviews_df = dfs["pr_reviews"].copy()
    if "pr_id" in reviews_df.columns:
        reviews_map = reviews_df.groupby("pr_id").size()
        reviewers_unique = reviews_df.groupby("pr_id")["user"].nunique()

        pr["num_reviews"] = pr["id"].map(reviews_map).fillna(0).astype(int)
        pr["num_reviewers_unique"] = pr["id"].map(reviewers_unique).fillna(0).astype(int)
    else:
        pr["num_reviews"] = 0
        pr["num_reviewers_unique"] = 0

    # Review comments v2
    rc = dfs["pr_review_comments_v2"].copy()
    if not rc.empty:
        rc["pr_number"] = _extract_pr_number_from_url(rc, "pull_request_url")
        rc_map = rc.groupby("pr_number").size()
        pr["num_review_comments"] = pr["number"].map(rc_map).fillna(0).astype(int)
    else:
        pr["num_review_comments"] = 0

    # =================
    # ISSUES ASSOCIÉES
    # =================
    rel = dfs["related_issue"].copy()

    if (not rel.empty) and ("pr_id" in rel.columns):
        linked_prs = set(rel["pr_id"].dropna().astype(int).unique())
        pr["has_issue_linked"] = pr["id"].isin(linked_prs).astype(int)
    else:
        pr["has_issue_linked"] = 0

    # ============
    # TEMPORALITÉ
    # ============
    pr["pr_duration_days"] = (pr["closed_at"] - pr["created_at"]).dt.total_seconds() / 86400
    pr["created_hour"] = pr["created_at"].dt.hour
    pr["merged_hour"] = pr["merged_at"].dt.hour

    # Nettoyage final
    pr.replace([np.inf, -np.inf], np.nan, inplace=True)
    pr = pr.fillna(0)

    return pr
