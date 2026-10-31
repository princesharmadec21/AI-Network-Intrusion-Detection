import joblib
import shap
import matplotlib.pyplot as plt
from src.config import TRAIN_FILE, MODEL_DIR, REPORT_DIR
from src.data_utils import load_dataset, binary_xy

def main():
    model_path = MODEL_DIR / "random_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train models first: python -m src.train")

    df = load_dataset(TRAIN_FILE)
    sample = df.sample(n=min(400, len(df)), random_state=42)
    X, _ = binary_xy(sample)

    pipeline = joblib.load(model_path)
    transformed = pipeline.named_steps["preprocess"].transform(X)
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(
        shap_values, transformed, feature_names=feature_names,
        show=False, max_display=15
    )
    plt.tight_layout()
    output = REPORT_DIR / "shap_summary.png"
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved: {output}")

if __name__ == "__main__":
    main()
