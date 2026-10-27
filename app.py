import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib", "Naive Bayes": "naive_bayes.joblib", "Support Vector Machine": "support_vector_machine.joblib"}

@st.cache_resource
def load_assets():
    # Streamlit Cloud starts from the GitHub repository. If model artifacts
    # were not committed, build them once from the bundled UCI dataset.
    if not (MODEL_DIR / "metrics.json").exists():
        from train_models import main as train_models
        with st.spinner("Preparing the trained models for first use..."):
            train_models()
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return metrics, {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}

def score(model, X, y):
    pred, prob = model.predict(X), model.predict_proba(X)
    return {"Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, prob, multi_class="ovr", average="macro"), "Precision": precision_score(y, pred, average="macro", zero_division=0), "Recall": recall_score(y, pred, average="macro", zero_division=0), "F1": f1_score(y, pred, average="macro", zero_division=0), "MCC": matthews_corrcoef(y, pred)}, pred

st.set_page_config(page_title="Dry Bean Classifier", page_icon="Bean", layout="wide")
st.title("UCI Dry Bean Classification - Five ML Models")
st.caption("13,611 bean images, 16 numerical features and 7 bean varieties. Macro-average metrics are used for this multi-class problem.")
metrics, models = load_assets(); model_name = st.selectbox("Select classification model", list(MODEL_FILES)); stored = metrics["models"][model_name]
st.subheader("Held-out test-set evaluation")
for col, metric in zip(st.columns(6), ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]): col.metric(metric, f"{stored[metric]:.4f}")
upload = st.file_uploader("Upload test data CSV (include the Class column for evaluation)", type="csv")
if upload:
    frame = pd.read_csv(upload); expected = metrics["feature_names"]; missing = set(expected) - set(frame.columns)
    if missing: st.error("Missing feature columns: " + ", ".join(sorted(missing)))
    else:
        X, model = frame[expected], models[model_name]
        if "Class" in frame.columns:
            live, pred = score(model, X, frame["Class"]); st.subheader("Uploaded test-data evaluation"); st.dataframe(pd.DataFrame([live]).style.format("{:.4f}"), use_container_width=True)
            cm = confusion_matrix(frame["Class"], pred, labels=metrics["class_labels"]); fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=metrics["class_labels"], yticklabels=metrics["class_labels"], ax=ax); ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix - {model_name}"); st.pyplot(fig)
            st.subheader("Classification Report")
            report = pd.DataFrame(classification_report(frame["Class"], pred, output_dict=True, zero_division=0)).transpose()
            report.index.name = "Class / Average"
            report_style = (
                report.style
                .format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"})
                .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"], vmin=0.0, vmax=1.0)
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#0B3C5D"), ("color", "white"), ("font-weight", "bold"), ("text-align", "center")]},
                    {"selector": "td", "props": [("text-align", "center"), ("border", "1px solid #D6E4F0")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#F4F9FD")]},
                ])
            )
            st.dataframe(report_style, use_container_width=True, height=360)
        else: frame["prediction"] = model.predict(X); st.info("No Class column was supplied, so predictions are displayed."); st.dataframe(frame[["prediction"]].head(20))
else:
    st.info("Upload the included test_data.csv to calculate metrics, a confusion matrix, and a classification report.")
    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(stored["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=metrics["class_labels"], yticklabels=metrics["class_labels"], ax=ax); ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix - {model_name}"); st.pyplot(fig)
