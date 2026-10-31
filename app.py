import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from src.config import MODEL_DIR, REPORT_DIR

st.set_page_config(page_title="AI-NIDS Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Powered Network Intrusion Detection")
st.caption("UNSW-NB15 | Offline ML research prototype")

binary_models = sorted([
    p for p in MODEL_DIR.glob("*.joblib")
    if p.stem != "multiclass_random_forest"
])
multi_path = MODEL_DIR / "multiclass_random_forest.joblib"

if not binary_models:
    st.warning("No trained models found. Run `python -m src.train` first.")
    st.stop()

tab_prediction, tab_analytics, tab_explain = st.tabs([
    "Traffic prediction", "Evaluation & analytics", "Explainable AI (SHAP)"
])

with tab_prediction:
    selected_model = st.selectbox(
        "Binary detection model", binary_models,
        format_func=lambda p: p.stem.replace("_", " ").title()
    )
    binary_model = joblib.load(selected_model)
    multi_model = joblib.load(multi_path) if multi_path.exists() else None

    uploaded = st.file_uploader(
        "Upload compatible UNSW-NB15 flow CSV", type=["csv"]
    )
    if uploaded is not None:
        raw = pd.read_csv(uploaded, low_memory=False)
        raw.columns = raw.columns.str.strip()
        X = raw.drop(columns=["id", "label", "attack_cat"], errors="ignore").copy()

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("string").str.strip().replace("", pd.NA).astype(object)
            numeric = pd.to_numeric(X[col], errors="coerce")
            if numeric.notna().mean() > 0.98:
                X[col] = numeric

        try:
            binary_pred = binary_model.predict(X)
            output = raw.copy()
            output["binary_prediction"] = binary_pred
            output["binary_result"] = pd.Series(binary_pred).map({0: "Normal", 1: "Attack"}).values

            if hasattr(binary_model, "predict_proba"):
                output["attack_probability"] = binary_model.predict_proba(X)[:, 1]

            if multi_model is not None:
                output["attack_category_prediction"] = multi_model.predict(X)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total flows", f"{len(output):,}")
            col2.metric("Predicted attacks", f"{int((binary_pred == 1).sum()):,}")
            col3.metric("Predicted normal", f"{int((binary_pred == 0).sum()):,}")

            st.plotly_chart(
                px.pie(output, names="binary_result", title="Predicted normal vs attack traffic"),
                width="stretch"
            )

            if "attack_category_prediction" in output.columns:
                attack_rows = output[output["binary_prediction"] == 1]
                if not attack_rows.empty:
                    counts = (
                        attack_rows["attack_category_prediction"]
                        .value_counts().rename_axis("category")
                        .reset_index(name="flows")
                    )
                    st.plotly_chart(
                        px.bar(counts, x="category", y="flows",
                               title="Predicted attack categories among predicted attacks"),
                        width="stretch"
                    )

            st.dataframe(output.head(100), width="stretch")
            st.download_button(
                "Download predictions CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="nids_predictions.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Upload a CSV with the same feature columns/schema as UNSW-NB15.")

with tab_analytics:
    comparison_file = REPORT_DIR / "model_comparison.csv"
    if comparison_file.exists():
        scores = pd.read_csv(comparison_file)
        st.subheader("Binary model comparison on official test set")
        st.dataframe(scores, width="stretch")
        st.plotly_chart(
            px.bar(scores, x="model", y=["precision", "recall", "f1"],
                   barmode="group", title="Precision, recall and F1-score"),
            width="stretch"
        )

        selected = st.selectbox("Select binary confusion matrix", scores["model"].tolist())
        cm_file = REPORT_DIR / f"{selected}_confusion_matrix.csv"
        if cm_file.exists():
            cm = pd.read_csv(cm_file, index_col=0)
            st.dataframe(cm, width="stretch")
            st.plotly_chart(
                px.imshow(cm, text_auto=True, aspect="auto",
                          title=f"{selected} confusion matrix",
                          labels=dict(x="Predicted", y="Actual", color="Count")),
                width="stretch"
            )
    else:
        st.info("Run `python -m src.evaluate` to generate evaluation analytics.")

    multi_metrics_file = REPORT_DIR / "multiclass_metrics.csv"
    if multi_metrics_file.exists():
        st.subheader("Multiclass attack-category metrics")
        st.dataframe(pd.read_csv(multi_metrics_file), width="stretch")

    multi_cm_file = REPORT_DIR / "multiclass_confusion_matrix.csv"
    if multi_cm_file.exists():
        st.subheader("Multiclass confusion matrix")
        multi_cm = pd.read_csv(multi_cm_file, index_col=0)
        st.dataframe(multi_cm, width="stretch")
        st.plotly_chart(
            px.imshow(multi_cm, text_auto=True, aspect="auto",
                      title="Multiclass confusion matrix",
                      labels=dict(x="Predicted category", y="Actual category", color="Count")),
            width="stretch"
        )

    report_file = REPORT_DIR / "multiclass_classification_report.txt"
    if report_file.exists():
        with st.expander("Per-category precision, recall and F1 report"):
            st.code(report_file.read_text(encoding="utf-8"))

with tab_explain:
    st.subheader("SHAP global feature explanation")
    shap_file = REPORT_DIR / "shap_summary.png"
    if shap_file.exists():
        st.image(str(shap_file), caption="Top features influencing binary Random Forest predictions")
    else:
        st.info("Run `python -m src.explain` in the terminal to generate the SHAP plot.")
    st.caption("SHAP explains model behaviour; feature importance does not prove causation.")

st.warning("Educational offline benchmark prototype only. Do not use as a production security control.")
