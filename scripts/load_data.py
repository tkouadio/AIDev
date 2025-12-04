from pathlib import Path
import pandas as pd

# Colonnes date pour PR
DATE_COLS_PR = ["created_at", "updated_at", "closed_at", "merged_at"]


def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Fichier manquant: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {path.name}: {e}")
        return pd.DataFrame()


def parse_dates(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_all(data_root: Path) -> dict:
   
    # FICHIERS PRIMAIRES
    pr_df   = read_parquet_safe(data_root / "pull_request.parquet")
    repo_df = read_parquet_safe(data_root / "repository.parquet")
    user_df = read_parquet_safe(data_root / "user.parquet")

    # COLLABORATION
    pr_comments_df        = read_parquet_safe(data_root / "pr_comments.parquet")
    pr_reviews_df         = read_parquet_safe(data_root / "pr_reviews.parquet")
    pr_review_comments_df = read_parquet_safe(data_root / "pr_review_comments_v2.parquet")

    # COMMITS
    pr_commits_df        = read_parquet_safe(data_root / "pr_commits.parquet")
    pr_commit_details_df = read_parquet_safe(data_root / "pr_commit_details.parquet")

    # ISSUES & TIMELINE
    related_issue_df = read_parquet_safe(data_root / "related_issue.parquet")
    issue_df         = read_parquet_safe(data_root / "issue.parquet")
    pr_timeline_df   = read_parquet_safe(data_root / "pr_timeline.parquet")
    pr_task_type_df  = read_parquet_safe(data_root / "pr_task_type.parquet")

    # DONNÉES HUMAINES
    human_pr_df           = read_parquet_safe(data_root / "human_pull_request.parquet")
    human_pr_task_type_df = read_parquet_safe(data_root / "human_pr_task_type.parquet")

    # ========================
    # NORMALISATION DES DATES
    # ========================

    # PR : created_at, updated_at, closed_at, merged_at
    pr_df = parse_dates(pr_df, DATE_COLS_PR)

    # Reviews / comments / timeline
    date_cols_general = ["created_at", "updated_at", "submitted_at"]

    for df in [pr_comments_df, pr_reviews_df, pr_review_comments_df, pr_timeline_df,
               pr_commits_df, pr_commit_details_df, issue_df]:
        for col in date_cols_general:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # ================================
    # RETOUR : structure standardisée
    # ================================

    return {
        "pr": pr_df,
        "repo": repo_df,
        "user": user_df,

        "pr_comments": pr_comments_df,
        "pr_reviews": pr_reviews_df,

        "pr_review_comments_v2": pr_review_comments_df,

        "pr_commits": pr_commits_df,
        "pr_commit_details": pr_commit_details_df,

        "related_issue": related_issue_df,
        "issue": issue_df,
        "pr_timeline": pr_timeline_df,
        "pr_task_type": pr_task_type_df,

        "human_pr": human_pr_df,
        "human_pr_task_type": human_pr_task_type_df,
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_ROOT = BASE_DIR / "data"

    dfs = load_all(DATA_ROOT)
    print({k: v.shape for k, v in dfs.items()})