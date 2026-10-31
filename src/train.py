import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import TRAIN_FILE, MODEL_DIR, REPORT_DIR, SAMPLE_FRAC, RANDOM_STATE
from src.data_utils import load_dataset, binary_xy, multiclass_xy
from src.models import build_preprocessor, binary_estimators, multiclass_estimator

def fit_and_save(name, X, y, estimator):
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("classifier", estimator)
    ])
    start = time.time()
    pipeline.fit(X, y)
    elapsed = time.time() - start
    joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
    print(f"Saved models/{name}.joblib | training time: {elapsed:.1f}s")
    return {"model": name, "training_seconds": round(elapsed, 2), "rows": len(X), "features": X.shape[1]}

def main():
    print("Loading UNSW-NB15 training dataset...")
    df = load_dataset(TRAIN_FILE)
    if SAMPLE_FRAC < 1.0:
        df = df.groupby("label", group_keys=False).sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    print(f"Training rows: {len(df):,}")

    X_binary, y_binary = binary_xy(df)
    summary = []
    for name, estimator in binary_estimators().items():
        print(f"\nTraining binary model: {name}")
        summary.append(fit_and_save(name, X_binary, y_binary, estimator))
    pd.DataFrame(summary).to_csv(REPORT_DIR / "training_summary.csv", index=False)

    print("\nPreparing multiclass attack categories...")
    X_multi, y_multi = multiclass_xy(df)
    print("Categories:", sorted(y_multi.unique().tolist()))
    print("Training multiclass Random Forest...")
    fit_and_save("multiclass_random_forest", X_multi, y_multi, multiclass_estimator())
    print("\nAll models trained successfully.")

if __name__ == "__main__":
    main()
