import pandas as pd

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Put UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv directly inside data/."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column. Available columns: {list(df.columns)}")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df

def clean_features(X):
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("string").str.strip().replace("", pd.NA).astype(object)
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notna().mean() > 0.98:
            X[col] = numeric
    return X

def binary_xy(df):
    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in ["id", "attack_cat", "label"] if c in df.columns], errors="ignore")
    return clean_features(X), y

def multiclass_xy(df):
    if "attack_cat" not in df.columns:
        raise ValueError("Missing 'attack_cat' column required for multiclass classification.")
    y = df["attack_cat"].astype("string").str.strip().fillna("")
    y = y.replace("", "Normal")
    y = y.where(df["label"].astype(int) == 1, "Normal")
    X = df.drop(columns=[c for c in ["id", "attack_cat", "label"] if c in df.columns], errors="ignore")
    return clean_features(X), y.astype(str)
