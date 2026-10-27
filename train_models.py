"""
Train, evaluate and save five machine-learning classifiers
using the UCI Dry Bean Dataset.

Models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

Outputs:
- Saved models in model/
- metrics.json
- metadata.json
- test_data.csv
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

SOURCE_FILE = DATA_DIR / "Dry_Bean_Dataset.xlsx"

TEST_DATA_FILE = ROOT / "test_data.csv"
METRICS_FILE = MODEL_DIR / "metrics.json"
METADATA_FILE = MODEL_DIR / "metadata.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate(model, X, y):
    """
    Evaluate a trained multiclass classification model.

    Metrics:
    - Accuracy
    - AUC
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient
    """

    predictions = model.predict(X)

    # All five models used in this project support predict_proba()
    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )

    return {
        "Accuracy": round(float(accuracy), 4),
        "AUC": round(float(auc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4)
    }


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("=" * 70)
print("DRY BEAN CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

print("\nChecking dataset...")

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        "\nDataset not found.\n"
        f"Expected file location:\n{SOURCE_FILE}\n\n"
        "Please copy Dry_Bean_Dataset.xlsx into the data folder."
    )

print(f"Dataset found: {SOURCE_FILE}")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_excel(SOURCE_FILE)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")


# ============================================================
# 6. BASIC DATA VALIDATION
# ============================================================

TARGET_COLUMN = "Class"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns: {list(df.columns)}"
    )


# Remove completely duplicated rows if present
duplicate_count = int(df.duplicated().sum())

print(f"\nDuplicate rows detected: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate rows removed.")


# Check missing values
missing_values = int(df.isnull().sum().sum())

print(f"Missing values detected: {missing_values}")

if missing_values > 0:
    print("Removing rows containing missing values...")
    df = df.dropna().reset_index(drop=True)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN]).copy()

y_original = df[TARGET_COLUMN].astype(str).copy()

FEATURE_NAMES = X.columns.tolist()

print("\nNumber of predictor features:", len(FEATURE_NAMES))

print("\nFeatures:")

for feature in FEATURE_NAMES:
    print(f" - {feature}")


# ============================================================
# 8. ENCODE TARGET CLASSES
# ============================================================

print("\nEncoding target classes...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_original)

CLASS_NAMES = label_encoder.classes_.tolist()

print("Classes detected:")

for number, class_name in enumerate(CLASS_NAMES):
    print(f" {number} -> {class_name}")

joblib.dump(
    label_encoder,
    LABEL_ENCODER_FILE
)

print(f"\nLabel encoder saved to: {LABEL_ENCODER_FILE}")


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\nCreating training and testing datasets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 10. DEFINE FIVE CLASSIFICATION MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]),

    # --------------------------------------------------------
    # Model 2: Decision Tree
    # --------------------------------------------------------

    "Decision Tree": Pipeline([
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]),

    # --------------------------------------------------------
    # Model 3: K-Nearest Neighbours
    # --------------------------------------------------------

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    # --------------------------------------------------------
    # Model 4: Gaussian Naive Bayes
    # --------------------------------------------------------

    "Naive Bayes": Pipeline([
        (
            "classifier",
            GaussianNB()
        )
    ]),

    # --------------------------------------------------------
    # Model 5: Support Vector Machine
    # --------------------------------------------------------
    #
    # CalibratedClassifierCV provides predict_proba()
    # without using the deprecated SVC(probability=True).

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    random_state=42
                ),
                method="sigmoid",
                cv=3,
                ensemble=False
            )
        )
    ])
}


# ============================================================
# 11. MODEL FILE NAMES
# ============================================================

model_file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# ============================================================
# 12. TRAIN MODELS
# ============================================================

results = {}

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Train model
    model.fit(
        X_train,
        y_train
    )

    print(f"{model_name} training completed.")

    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    model_results = evaluate(
        model,
        X_test,
        y_test
    )

    results[model_name] = model_results

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / model_file_names[model_name]

    joblib.dump(
        model,
        model_path
    )

    print(f"Saved: {model_path}")

    # --------------------------------------------------------
    # Display test results
    # --------------------------------------------------------

    print("Test Results:")

    for metric_name, metric_value in model_results.items():
        print(f"  {metric_name:<12}: {metric_value}")


# ============================================================
# 13. SAVE METRICS
# ============================================================

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

print(f"\nMetrics saved to: {METRICS_FILE}")


# ============================================================
# 14. SAVE METADATA
# ============================================================

metadata = {
    "target_column": TARGET_COLUMN,
    "feature_names": FEATURE_NAMES,
    "class_names": CLASS_NAMES,

    "number_of_instances": int(len(df)),
    "number_of_features": int(len(FEATURE_NAMES)),
    "number_of_classes": int(len(CLASS_NAMES)),

    "model_files": model_file_names
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

print(f"Metadata saved to: {METADATA_FILE}")