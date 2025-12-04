import pandas as pd


def merge_with_user_repo(pr: pd.DataFrame, user_df: pd.DataFrame, repo_df: pd.DataFrame):

    # ===========
    # USER MERGE
    # ===========
    user = user_df.copy()
    user.rename(columns={"id": "user_id"}, inplace=True)

    if "created_at" in user.columns:
        user["created_at"] = pd.to_datetime(user["created_at"], errors="coerce")

    keep_user_cols = []
    for c in ["user_id", "created_at", "followers", "public_repos"]:
        if c in user.columns:
            keep_user_cols.append(c)

    pr = pr.merge(
        user[keep_user_cols],
        on="user_id",
        how="left",
        suffixes=("", "_user")
    )

    # Calcul author tenure
    if "created_at_user" in pr.columns:
        pr["author_tenure_days"] = (
            pr["created_at"] - pr["created_at_user"]
        ).dt.total_seconds() / 86400
    else:
        pr["author_tenure_days"] = 0

    pr["author_tenure_days"] = pr["author_tenure_days"].fillna(0)

    # ===========
    # REPO MERGE
    # ===========
    repo = repo_df.copy()
    repo.rename(columns={"id": "repo_id"}, inplace=True)

    rename_map = {
        "forks_count": "forks",
        "stargazers_count": "stars",
    }
    for old, new in rename_map.items():
        if old in repo.columns and new not in repo.columns:
            repo[new] = repo[old]

    repo_cols = []
    for c in ["repo_id", "language", "forks", "stars", "description"]:
        if c in repo.columns:
            repo_cols.append(c)

    pr = pr.merge(
        repo[repo_cols],
        on="repo_id",
        how="left"
    )

    # Remplissage des valeurs vides
    for col in ["forks", "stars"]:
        if col in pr.columns:
            pr[col] = pr[col].fillna(0)

    # Longueur description du repo
    if "description" in pr.columns:
        pr["repo_description_length"] = pr["description"].fillna("").str.len()
    else:
        pr["repo_description_length"] = 0

    return pr