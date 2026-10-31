import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from src.config import TEST_FILE, MODEL_DIR, REPORT_DIR
from src.data_utils import load_dataset, binary_xy, multiclass_xy

def main():
    df = load_dataset(TEST_FILE)
    results = []

    X_test, y_test = binary_xy(df)
    binary_paths = [p for p in MODEL_DIR.glob("*.joblib") if p.stem != "multiclass_random_forest"]
    if not binary_paths:
        raise FileNotFoundError("No trained models found. Run: python -m src.train")

    for path in sorted(binary_paths):
        model = joblib.load(path)
        predictions = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        row = {
            "model": path.stem,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0
        }
        results.append(row)
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual normal", "Actual attack"],
            columns=["Predicted normal", "Predicted attack"]
        )
        cm.to_csv(REPORT_DIR / f"{path.stem}_confusion_matrix.csv")
        (REPORT_DIR / f"{path.stem}_classification_report.txt").write_text(
            classification_report(y_test, predictions, target_names=["Normal", "Attack"], zero_division=0),
            encoding="utf-8"
        )
        print(row)

    pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )

    multi_path = MODEL_DIR / "multiclass_random_forest.joblib"
    if multi_path.exists():
        X_multi, y_multi = multiclass_xy(df)
        multi_model = joblib.load(multi_path)
        multi_pred = multi_model.predict(X_multi)
        labels = sorted(set(y_multi.unique()) | set(multi_pred))
        cm_multi = confusion_matrix(y_multi, multi_pred, labels=labels)
        pd.DataFrame(
            cm_multi,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels]
        ).to_csv(REPORT_DIR / "multiclass_confusion_matrix.csv")
        (REPORT_DIR / "multiclass_classification_report.txt").write_text(
            classification_report(y_multi, multi_pred, labels=labels, zero_division=0),
            encoding="utf-8"
        )
        pd.DataFrame([{
            "accuracy": accuracy_score(y_multi, multi_pred),
            "macro_precision": precision_score(y_multi, multi_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_multi, multi_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_multi, multi_pred, average="macro", zero_division=0)
        }]).to_csv(REPORT_DIR / "multiclass_metrics.csv", index=False)
        print("Multiclass evaluation complete.")

    print(f"\nReports saved in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
