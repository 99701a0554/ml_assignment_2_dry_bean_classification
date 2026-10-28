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

TRAIN_DATA_FILE = DATA_DIR / "training_data.csv"
TEST_DATA_FILE = ROOT / "test_data.csv"
METRICS_FILE = MODEL_DIR / "metrics.json"
METADATA_FILE = MODEL_DIR / "metadata.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"

README_FILE = ROOT / "README.md"

README_AUTO_START = (
    "<!-- AUTO-GENERATED-RESULTS:START -->"
)

README_AUTO_END = (
    "<!-- AUTO-GENERATED-RESULTS:END -->"
)


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
# README UPDATE FUNCTION
# ============================================================

def update_readme_results(
    readme_file,
    metadata,
    results
):
    """
    Update only the protected auto-generated section
    in README.md.

    Manual README content outside the START and END
    markers is preserved.
    """

    # --------------------------------------------------------
    # Validate README
    # --------------------------------------------------------

    if not readme_file.exists():

        print(
            f"\nREADME not found: {readme_file}"
        )

        print(
            "README automatic update skipped."
        )

        return


    existing_content = readme_file.read_text(
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Validate protected markers
    # --------------------------------------------------------

    start_position = existing_content.find(
        README_AUTO_START
    )

    end_position = existing_content.find(
        README_AUTO_END
    )


    if start_position == -1:

        raise ValueError(
            "README START marker was not found:\n"
            f"{README_AUTO_START}"
        )


    if end_position == -1:

        raise ValueError(
            "README END marker was not found:\n"
            f"{README_AUTO_END}"
        )


    if end_position <= start_position:

        raise ValueError(
            "README automatic-results markers "
            "are in the wrong order."
        )


    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    original_records = int(
        metadata.get(
            "original_number_of_instances",
            0
        )
    )


    duplicate_records = int(
        metadata.get(
            "duplicate_records_detected",
            0
        )
    )


    model_ready_records = int(
        metadata.get(
            "number_of_instances",
            0
        )
    )


    training_records = int(
        metadata.get(
            "training_records",
            0
        )
    )


    test_records = int(
        metadata.get(
            "test_records",
            0
        )
    )


    number_of_features = int(
        metadata.get(
            "number_of_features",
            0
        )
    )


    number_of_classes = int(
        metadata.get(
            "number_of_classes",
            0
        )
    )


    class_names = metadata.get(
        "class_names",
        []
    )


    # --------------------------------------------------------
    # Calculate train/test percentages
    # --------------------------------------------------------

    if model_ready_records > 0:

        training_percentage = (
            training_records
            / model_ready_records
            * 100
        )

        test_percentage = (
            test_records
            / model_ready_records
            * 100
        )

    else:

        training_percentage = 0.0
        test_percentage = 0.0


    # --------------------------------------------------------
    # Prepare results DataFrame
    # --------------------------------------------------------

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]


    results_df = pd.DataFrame(
        results
    ).T


    results_df = results_df.reindex(
        columns=metric_names
    )


    results_df = results_df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    if results_df.empty:

        raise ValueError(
            "No model results are available "
            "for README generation."
        )


    # --------------------------------------------------------
    # Overall winner based on weighted F1
    # --------------------------------------------------------

    valid_f1 = results_df[
        "F1"
    ].dropna()


    if valid_f1.empty:

        raise ValueError(
            "F1 results are unavailable. "
            "README winner cannot be calculated."
        )


    overall_winner = valid_f1.idxmax()


    winner_metrics = results_df.loc[
        overall_winner
    ]


    # --------------------------------------------------------
    # Find best model for every metric
    # --------------------------------------------------------

    best_models = {}


    for metric in metric_names:

        valid_scores = results_df[
            metric
        ].dropna()


        if not valid_scores.empty:

            best_model = valid_scores.idxmax()

            best_models[metric] = {
                "model": best_model,
                "score": valid_scores.loc[
                    best_model
                ],
            }


    # --------------------------------------------------------
    # DATASET SUMMARY TABLE
    # --------------------------------------------------------

    dataset_table = f"""
| Dataset Measure | Value |
|---|---:|
| Original records | {original_records:,} |
| Duplicate records removed | {duplicate_records:,} |
| Model-ready records | {model_ready_records:,} |
| Training records | {training_records:,} ({training_percentage:.1f}%) |
| Test records | {test_records:,} ({test_percentage:.1f}%) |
| Numerical features | {number_of_features} |
| Bean classes | {number_of_classes} |
"""


    # --------------------------------------------------------
    # MODEL PERFORMANCE TABLE
    # --------------------------------------------------------

    comparison_lines = [
        (
            "| Model | Accuracy | AUC | Precision | "
            "Recall | F1 | MCC |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]


    for model_name, row in results_df.iterrows():

        comparison_lines.append(
            f"| {model_name} "
            f"| {row['Accuracy']:.4f} "
            f"| {row['AUC']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['MCC']:.4f} |"
        )


    comparison_table = "\n".join(
        comparison_lines
    )


    # --------------------------------------------------------
    # BEST MODEL BY METRIC TABLE
    # --------------------------------------------------------

    winner_lines = [
        "| Metric | Best Model | Score |",
        "|---|---|---:|",
    ]


    for metric in metric_names:

        if metric in best_models:

            winner_lines.append(
                f"| {metric} "
                f"| {best_models[metric]['model']} "
                f"| {best_models[metric]['score']:.4f} |"
            )


    winner_table = "\n".join(
        winner_lines
    )


    # --------------------------------------------------------
    # BEAN CLASS LIST
    # --------------------------------------------------------

    class_list = "\n".join(
        f"- {class_name}"
        for class_name in class_names
    )


    # --------------------------------------------------------
    # PERFORMANCE OBSERVATIONS
    # --------------------------------------------------------

    observations = []


    metric_labels = {
        "Accuracy": "Accuracy",
        "AUC": "AUC",
        "Precision": "Weighted Precision",
        "Recall": "Weighted Recall",
        "F1": "Weighted F1 Score",
        "MCC": (
            "Matthews Correlation "
            "Coefficient (MCC)"
        ),
    }


    for metric in metric_names:

        if metric in best_models:

            model_name = best_models[
                metric
            ]["model"]

            score_value = best_models[
                metric
            ]["score"]


            observations.append(
                f"- **{metric_labels[metric]}:** "
                f"{model_name} achieved the highest "
                f"score of **{score_value:.4f}**."
            )


    observations_text = "\n".join(
        observations
    )
    # --------------------------------------------------------
    # DYNAMIC MODEL-WISE PERFORMANCE OBSERVATIONS
    # Assignment requirement - 3 marks
    # --------------------------------------------------------

    def build_model_observation(
        model_name,
        row,
        results_df,
        overall_winner
    ):
        """
        Build model-specific observations entirely from
        the current training/evaluation results.

        No metric values, rankings or winner names
        are hard-coded.
        """

        # ----------------------------------------------------
        # Rank models dynamically by weighted F1
        # ----------------------------------------------------

        f1_ranking = (
            results_df["F1"]
            .rank(
                ascending=False,
                method="min"
            )
        )

        model_rank = int(
            f1_ranking.loc[model_name]
        )

        total_models = len(
            results_df
        )

        # ----------------------------------------------------
        # Find which metrics this model leads
        # ----------------------------------------------------

        metrics_won = []

        for metric in metric_names:

            valid_scores = (
                results_df[metric]
                .dropna()
            )

            if valid_scores.empty:
                continue

            if (
                row[metric]
                == valid_scores.max()
            ):

                metric_label = (
                    "Weighted F1"
                    if metric == "F1"
                    else metric
                )

                metrics_won.append(
                    metric_label
                )

        # ----------------------------------------------------
        # Difference from winning F1
        # ----------------------------------------------------

        winning_f1 = float(
            results_df.loc[
                overall_winner,
                "F1"
            ]
        )

        model_f1 = float(
            row["F1"]
        )

        f1_difference = (
            winning_f1
            - model_f1
        )

        # ----------------------------------------------------
        # Dynamic ranking observation
        # ----------------------------------------------------

        if model_name == overall_winner:

            rank_observation = (
                "This model ranked first by weighted F1 "
                "and is selected as the overall winner."
            )

        elif model_rank == 2:

            rank_observation = (
                "This model ranked second by weighted F1. "
                f"Its F1 score was {f1_difference:.4f} "
                "below the winning model."
            )

        elif model_rank == 3:

            rank_observation = (
                "This model ranked third by weighted F1. "
                f"Its F1 score was {f1_difference:.4f} "
                "below the winning model."
            )

        else:

            rank_observation = (
                f"This model ranked {model_rank} out of "
                f"{total_models} models by weighted F1. "
                f"Its F1 score was {f1_difference:.4f} "
                "below the winning model."
            )

        # ----------------------------------------------------
        # Dynamic metric-leadership observation
        # ----------------------------------------------------

        if metrics_won:

            metric_observation = (
                " It achieved the highest score for "
                + ", ".join(metrics_won)
                + "."
            )

        else:

            metric_observation = (
                " It did not achieve the highest score "
                "for any of the evaluated metrics."
            )

        # ----------------------------------------------------
        # Final model observation
        # ----------------------------------------------------

        return (
            f"Accuracy **{row['Accuracy']:.4f}**, "
            f"AUC **{row['AUC']:.4f}**, "
            f"Precision **{row['Precision']:.4f}**, "
            f"Recall **{row['Recall']:.4f}**, "
            f"Weighted F1 **{row['F1']:.4f}**, "
            f"and MCC **{row['MCC']:.4f}**. "
            f"{rank_observation}"
            f"{metric_observation}"
        )


    # --------------------------------------------------------
    # Build assignment observation table dynamically
    # --------------------------------------------------------

    model_observation_lines = [
        (
            "| ML Model Name | "
            "Observation about Model Performance |"
        ),
        "|---|---|",
    ]


    for model_name, row in results_df.iterrows():

        display_name = (
            "Support Vector Machine"
            if model_name == "SVM"
            else model_name
        )

        observation = build_model_observation(
            model_name=model_name,
            row=row,
            results_df=results_df,
            overall_winner=overall_winner
        )

        model_observation_lines.append(
            f"| **{display_name}** "
            f"| {observation} |"
        )


    # --------------------------------------------------------
    # Overall-winner information entirely from results
    # --------------------------------------------------------

    winner_display_name = (
        "Support Vector Machine"
        if overall_winner == "SVM"
        else overall_winner
    )

    winner_metric_names = []


    for metric in metric_names:

        valid_scores = (
            results_df[metric]
            .dropna()
        )

        if valid_scores.empty:
            continue

        if (
            winner_metrics[metric]
            == valid_scores.max()
        ):

            metric_label = (
                "Weighted F1"
                if metric == "F1"
                else metric
            )

            winner_metric_names.append(
                metric_label
            )


    if winner_metric_names:

        winner_metric_text = (
            "The model also achieved the highest score "
            "for "
            + ", ".join(winner_metric_names)
            + "."
        )

    else:

        winner_metric_text = (
            "The winner was selected using the "
            "weighted F1 criterion."
        )


    model_observation_lines.append(
        "| **Overall Winner for the Dataset** | "
        f"**{winner_display_name}** is selected as the "
        f"overall winner because it achieved the highest "
        f"weighted F1 score of "
        f"**{winner_metrics['F1']:.4f}**. "
        f"{winner_metric_text} |"
    )


    model_observation_table = "\n".join(
        model_observation_lines
    )

    # --------------------------------------------------------
    # BUILD GENERATED README SECTION
    # --------------------------------------------------------

    generated_content = f"""
{README_AUTO_START}

## Latest Generated Results

> This section is generated automatically by
> `train_models.py`. Manual content outside these
> markers is preserved.

### Dataset Summary

{dataset_table}

The original dataset contained **{original_records:,}**
records. After preprocessing,
**{model_ready_records:,}** records were available
for modelling.

The model-ready dataset was divided into:

- **{training_records:,} training records**
  ({training_percentage:.1f}%)
- **{test_records:,} test records**
  ({test_percentage:.1f}%)

### Bean Classes

{class_list}

### Model Performance Comparison

The models were evaluated on the same held-out
test dataset.

Weighted Precision, Recall and F1 are used for
multiclass evaluation.

{comparison_table}

### Best Model by Metric

{winner_table}

### Model Performance Observations

{observations_text}

### Model-wise Performance Observations

The following observations are calculated dynamically from
the latest held-out test-set evaluation.

{model_observation_table}

### Overall Winner

**{winner_display_name}** is selected as the overall
winning model based on the highest weighted F1
score of **{winner_metrics['F1']:.4f}**.

The overall winner achieved:

- Accuracy: **{winner_metrics['Accuracy']:.4f}**
- AUC: **{winner_metrics['AUC']:.4f}**
- Precision: **{winner_metrics['Precision']:.4f}**
- Recall: **{winner_metrics['Recall']:.4f}**
- Weighted F1: **{winner_metrics['F1']:.4f}**
- MCC: **{winner_metrics['MCC']:.4f}**

Weighted F1 is used as the primary selection
criterion because it combines Precision and Recall
while accounting for class support.

{README_AUTO_END}
"""


    # --------------------------------------------------------
    # REPLACE ONLY GENERATED SECTION
    # --------------------------------------------------------

    end_position = (
        end_position
        + len(README_AUTO_END)
    )


    updated_content = (
        existing_content[
            :start_position
        ]
        + generated_content.strip()
        + existing_content[
            end_position:
        ]
    )


    # --------------------------------------------------------
    # WRITE UPDATED README
    # --------------------------------------------------------

    readme_file.write_text(
        updated_content,
        encoding="utf-8"
    )


    print(
        "\nREADME automatic results "
        "updated successfully."
    )

    print(
        f"README file: {readme_file}"
    )

    print(
        f"Overall winner: {overall_winner}"
    )

    print(
        f"Winning weighted F1: "
        f"{winner_metrics['F1']:.4f}"
    )

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

# Record count before any data cleaning
ORIGINAL_NUMBER_OF_INSTANCES = len(df)

#PROCESSED_NUMBER_OF_INSTANCES = len(df)

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

# Record count after data cleaning
PROCESSED_NUMBER_OF_INSTANCES = len(df)

print(
    f"Original records: {ORIGINAL_NUMBER_OF_INSTANCES:,}"
)

print(
    f"Processed records: {PROCESSED_NUMBER_OF_INSTANCES:,}"
)

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

# Actual train/test record counts
TRAINING_RECORDS = len(X_train)
TEST_RECORDS = len(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# UPDATE EXISTING TRAINING CSV
# ============================================================

training_data = X_train.copy()

training_data[TARGET_COLUMN] = (
    label_encoder.inverse_transform(y_train)
)

training_data.to_csv(
    TRAIN_DATA_FILE,
    index=False
)

print(
    f"Training CSV updated: {TRAIN_DATA_FILE}"
)

print(
    f"Training records: {TRAINING_RECORDS:,}"
)


# ============================================================
# UPDATE EXISTING TEST CSV
# ============================================================

test_data = X_test.copy()

test_data[TARGET_COLUMN] = (
    label_encoder.inverse_transform(y_test)
)

test_data.to_csv(
    TEST_DATA_FILE,
    index=False
)

print(
    f"Test CSV updated: {TEST_DATA_FILE}"
)

print(
    f"Test records: {TEST_RECORDS:,}"
)


# ============================================================
# VALIDATE SPLIT
# ============================================================

assert (
    TRAINING_RECORDS + TEST_RECORDS
    == len(df)
), (
    "Training + Test record counts do not match "
    "the processed dataset count."
)

print(
    f"Total records split: "
    f"{TRAINING_RECORDS + TEST_RECORDS:,}"
)

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

    "original_number_of_instances": int(
        ORIGINAL_NUMBER_OF_INSTANCES
    ),

    "duplicate_records_detected": int(
    duplicate_count
    ),

    "number_of_instances": int(
    PROCESSED_NUMBER_OF_INSTANCES
   ),

    "training_records": int(
        TRAINING_RECORDS
    ),

    "test_records": int(
        TEST_RECORDS
    ),

    "test_size": 0.20,

    "random_state": 42,

    "number_of_features": int(
        len(FEATURE_NAMES)
    ),

    "number_of_classes": int(
        len(CLASS_NAMES)
    ),

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

# ============================================================
# 15. UPDATE README
# ============================================================

update_readme_results(
    readme_file=README_FILE,
    metadata=metadata,
    results=results
)