"""Train and save the five required classifiers on the UCI Dry Bean Dataset."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
SOURCE_FILE = ROOT / "data" / "Dry_Bean_Dataset.xlsx"

def evaluate(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    return {"Accuracy": round(float(accuracy_score(y, prediction)), 4), "AUC": round(float(roc_auc_score(y, probability, multi_class="ovr", average="macro")), 4), "Precision": round(float(precision_score(y, prediction, average="macro", zero_division=0)), 4), "Recall": round(float(recall_score(y, prediction, average="macro", zero_division=0)), 4), "F1": round(float(f1_score(y, prediction, average="macro", zero_division=0)), 4), "MCC": round(float(matthews_corrcoef(y, prediction)), 4), "confusion_matrix": confusion_matrix(y, prediction, labels=model.classes_).tolist(), "classification_report": classification_report(y, prediction, labels=model.classes_, output_dict=True, zero_division=0)}

def main():
    MODEL_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
    if not SOURCE_FILE.exists(): raise FileNotFoundError(f"Dataset not found: {SOURCE_FILE}")
    data = pd.read_excel(SOURCE_FILE)
    X, y = data.drop(columns=["Class"]), data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    pd.concat([X_test.reset_index(drop=True), y_test.rename("Class").reset_index(drop=True)], axis=1).to_csv(ROOT / "test_data.csv", index=False)
    pd.concat([X_train.reset_index(drop=True), y_train.rename("Class").reset_index(drop=True)], axis=1).to_csv(DATA_DIR / "training_data.csv", index=False)
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, C=2.0, random_state=42))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, min_samples_leaf=2, random_state=42),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=5.0, probability=True, random_state=42))]),
    }
    results = {"dataset": "UCI Dry Bean Dataset", "test_size": int(len(X_test)), "feature_names": list(X.columns), "class_labels": sorted(y.unique().tolist()), "metric_note": "AUC, precision, recall and F1 are macro-averaged; AUC uses one-vs-rest multiclass ROC-AUC.", "models": {}}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        results["models"][name] = evaluate(model, X_test, y_test)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
