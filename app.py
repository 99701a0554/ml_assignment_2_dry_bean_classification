import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Assignment - UCI Dry Bean Classification",
    page_icon="🌱",
    layout="wide",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

METRICS_FILE = MODEL_DIR / "metrics.json"
METADATA_FILE = MODEL_DIR / "metadata.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"


# ============================================================
# 3. FALLBACK MODEL FILE NAMES
# ============================================================

DEFAULT_MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}


# ============================================================
# 4. LOAD APPLICATION ASSETS
# ============================================================

#@st.cache_resource
def load_assets():

    required_files = [
        METRICS_FILE,
        METADATA_FILE,
        LABEL_ENCODER_FILE,
    ]

    missing_files = [
        file for file in required_files
        if not file.exists()
    ]

    if missing_files:

        missing_list = "\n".join(
            f"- {file.name}"
            for file in missing_files
        )

        raise FileNotFoundError(
            "Required model files are missing:\n\n"
            + missing_list
            + "\n\nRun train_models.py and make sure "
              "the generated files are available in model/."
        )

    # Load metrics
    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    # Load metadata
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # Load label encoder
    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    # Model filenames come from metadata.json
    model_files = metadata.get(
        "model_files",
        DEFAULT_MODEL_FILES,
    )

    models = {}

    for model_name, file_name in model_files.items():

        model_path = MODEL_DIR / file_name

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path.name}"
            )

        models[model_name] = joblib.load(
            model_path
        )

    return (
        metrics,
        metadata,
        label_encoder,
        models,
    )


# ============================================================
# 5. MODEL SCORING FUNCTION
# ============================================================

def score(model, X, y):

    pred = model.predict(X)
    prob = model.predict_proba(X)

    results = {

        "Accuracy": accuracy_score(
            y,
            pred,
        ),

        "AUC": roc_auc_score(
            y,
            prob,
            multi_class="ovr",
            average="weighted",
        ),

        "Precision": precision_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y,
            pred,
        ),
    }

    return results, pred


# ============================================================
# 6. CLASSIFICATION REPORT STYLING
# ============================================================

def style_report_table(table, index_name):

    table = table.copy()
    table.index.name = index_name

    return (
        table.style

        .format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }
        )

        .background_gradient(
            cmap="Blues",
            subset=[
                "precision",
                "recall",
                "f1-score",
            ],
            vmin=0.0,
            vmax=1.0,
        )

        .set_properties(
            subset=["support"],
            **{
                "background-color": "#DDEBF7",
                "color": "#0B3C5D",
                "font-weight": "bold",
            },
        )

        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0B3C5D"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ],
                },

                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("border", "1px solid #D6E4F0"),
                    ],
                },

                {
                    "selector": "tbody tr:nth-child(even)",
                    "props": [
                        ("background-color", "#F4F9FD"),
                    ],
                },
            ]
        )
    )


# ============================================================
# 7. LOAD FILES
# ============================================================

try:

    (
        metrics,
        metadata,
        label_encoder,
        models,
    ) = load_assets()

except Exception as error:

    st.error(
        f"Unable to load application assets:\n\n{error}"
    )

    st.stop()


# ============================================================
# 8. READ METADATA
# ============================================================

FEATURE_NAMES = metadata["feature_names"]
CLASS_NAMES = metadata["class_names"]
TARGET_COLUMN = metadata["target_column"]


# ------------------------------------------------------------
# Safely read integer metadata values
# ------------------------------------------------------------

def get_metadata_int(key, default=None):
    """
    Safely retrieve an integer value from metadata.
    Returns default when the key is missing, None,
    empty, or cannot be converted to an integer.
    """

    value = metadata.get(key)

    if value is None or value == "":
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------

ORIGINAL_NUMBER_OF_INSTANCES = get_metadata_int(
    "original_number_of_instances"
)

DUPLICATE_RECORDS_DETECTED = get_metadata_int(
    "duplicate_records_detected",
    0
)

NUMBER_OF_INSTANCES = get_metadata_int(
    "number_of_instances"
)

TRAINING_RECORDS = get_metadata_int(
    "training_records"
)

TEST_RECORDS = get_metadata_int(
    "test_records"
)

NUMBER_OF_FEATURES = get_metadata_int(
    "number_of_features",
    len(FEATURE_NAMES)
)

NUMBER_OF_CLASSES = get_metadata_int(
    "number_of_classes",
    len(CLASS_NAMES)
)

# ============================================================
# 9. APPLICATION TITLE
# ============================================================

st.title(
    "🌱 Machine Learning Assignment: UCI Dry Bean Classification Using Five ML Models"
)


# ------------------------------------------------------------
# Dataset caption
# ------------------------------------------------------------

if (
    NUMBER_OF_INSTANCES is not None
    and ORIGINAL_NUMBER_OF_INSTANCES is not None
    and ORIGINAL_NUMBER_OF_INSTANCES != NUMBER_OF_INSTANCES
):

    st.caption(
    (
        f"{NUMBER_OF_INSTANCES:,} bean instances, "
        if NUMBER_OF_INSTANCES is not None
        else ""
    )
    + f"{NUMBER_OF_FEATURES} numerical features and "
    + f"{NUMBER_OF_CLASSES} bean varieties. "
    + "Weighted-average metrics are used for this "
    + "multi-class classification problem."
)

elif NUMBER_OF_INSTANCES is not None:

    st.caption(
        f"{NUMBER_OF_INSTANCES:,} bean instances, "
        f"{NUMBER_OF_FEATURES} numerical features and "
        f"{NUMBER_OF_CLASSES} bean varieties. "
        "Weighted-average metrics are used for this "
        "multi-class classification problem."
    )

else:

    st.caption(
        f"{NUMBER_OF_FEATURES} numerical features and "
        f"{NUMBER_OF_CLASSES} bean varieties. "
        "Weighted-average metrics are used for this "
        "multi-class classification problem."
    )
# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown("#### Dataset Summary")


# ------------------------------------------------------------
# Calculate training / testing percentages
# ------------------------------------------------------------

TRAIN_PERCENTAGE = None
TEST_PERCENTAGE = None

if (
    NUMBER_OF_INSTANCES is not None
    and NUMBER_OF_INSTANCES > 0
):

    if TRAINING_RECORDS is not None:

        TRAIN_PERCENTAGE = (
            TRAINING_RECORDS
            / NUMBER_OF_INSTANCES
            * 100
        )

    if TEST_RECORDS is not None:

        TEST_PERCENTAGE = (
            TEST_RECORDS
            / NUMBER_OF_INSTANCES
            * 100
        )


# ------------------------------------------------------------
# Dataset Processing Summary
# ------------------------------------------------------------

st.markdown("##### Data Preparation")

original_col, duplicate_col, ready_col = st.columns(3)


original_col.metric(
    "Original Records",
    (
        f"{ORIGINAL_NUMBER_OF_INSTANCES:,}"
        if ORIGINAL_NUMBER_OF_INSTANCES is not None
        else "N/A"
    )
)


duplicate_col.metric(
    "Duplicate Records Removed",
    (
        f"{DUPLICATE_RECORDS_DETECTED:,}"
        if DUPLICATE_RECORDS_DETECTED is not None
        else "N/A"
    )
)


ready_col.metric(
    "Model-ready Records",
    (
        f"{NUMBER_OF_INSTANCES:,}"
        if NUMBER_OF_INSTANCES is not None
        else "N/A"
    )
)


# ------------------------------------------------------------
# Train / Test Split
# ------------------------------------------------------------

st.markdown("##### Train / Test Split")

training_col, testing_col = st.columns(2)


training_col.metric(
    "Training Records",
    (
        f"{TRAINING_RECORDS:,}"
        if TRAINING_RECORDS is not None
        else "N/A"
    ),
    (
        f"{TRAIN_PERCENTAGE:.1f}% of model-ready data"
        if TRAIN_PERCENTAGE is not None
        else None
    ),
    delta_color="off"
)


testing_col.metric(
    "Test Records",
    (
        f"{TEST_RECORDS:,}"
        if TEST_RECORDS is not None
        else "N/A"
    ),
    (
        f"{TEST_PERCENTAGE:.1f}% of model-ready data"
        if TEST_PERCENTAGE is not None
        else None
    ),
    delta_color="off"
)


# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

if (
    NUMBER_OF_INSTANCES is not None
    and TRAINING_RECORDS is not None
    and TEST_RECORDS is not None
):

    if (
        TRAINING_RECORDS + TEST_RECORDS
        != NUMBER_OF_INSTANCES
    ):

        st.warning(
            "Training + Test records do not equal "
            "the model-ready dataset size."
        )


    # --------------------------------------------------------
    # Integrity validation
    # --------------------------------------------------------

    if (
        TRAINING_RECORDS + TEST_RECORDS
        != NUMBER_OF_INSTANCES
    ):

        st.warning(
            "Training and test record counts do not match "
            "the model-ready dataset size."
        )

else:

    st.info(
        "Training/test split information is not available "
        "in metadata.json."
    )

# ============================================================
# 10. DATASET INFORMATION
# ============================================================

with st.expander(
    "Dataset and Model Information"
):

    st.write(
        f"**Target column:** {TARGET_COLUMN}"
    )

    st.write(
        f"**Number of features:** {NUMBER_OF_FEATURES}"
    )

    st.write(
        f"**Number of classes:** {NUMBER_OF_CLASSES}"
    )

    st.write(
        "**Class labels:**"
    )

    for class_name in CLASS_NAMES:

        st.write(
            f"- {class_name}"
        )


# ============================================================
# 11. MODEL SELECTION
# ============================================================

st.subheader(
    "1. Select Classification Model"
)

model_name = st.selectbox(
    "Select classification model",
    list(models.keys()),
)

model = models[model_name]


# ============================================================
# 12. STORED MODEL METRICS
# ============================================================

if model_name not in metrics:

    st.error(
        f"No evaluation metrics are available for "
        f"{model_name}."
    )

    st.stop()


stored = metrics[model_name]


# ============================================================
# 13. HELD-OUT TEST-SET EVALUATION
# ============================================================

st.subheader(
    "2. Held-out Test-set Evaluation"
)

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

metric_columns = st.columns(6)


for column, metric in zip(
    metric_columns,
    metric_names,
):

    value = stored.get(metric)

    if value is not None:

        column.metric(
            metric,
            f"{float(value):.4f}",
        )

    else:

        column.metric(
            metric,
            "N/A",
        )


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

st.subheader(
    "3. Comparison of All Five Models"
)

performance_df = (
    pd.DataFrame(metrics)
    .T
    .reindex(columns=metric_names)
)

performance_df = performance_df.apply(
    pd.to_numeric,
    errors="coerce",
)


st.dataframe(
    performance_df.style
    .format("{:.4f}")
    .background_gradient(
        cmap="Blues",
        subset=metric_names,
        vmin=0,
        vmax=1,
    ),
    use_container_width=True,
)


# ============================================================
# 15. PERFORMANCE OBSERVATIONS
# ============================================================

st.subheader(
    "4. Model Performance Observations"
)


# ------------------------------------------------------------
# Find best model for each metric
# ------------------------------------------------------------

best_models = {}

for metric in metric_names:

    valid_values = performance_df[
        metric
    ].dropna()

    if not valid_values.empty:

        best_model = valid_values.idxmax()

        best_models[metric] = {
            "model": best_model,
            "score": valid_values.loc[
                best_model
            ],
        }


# ------------------------------------------------------------
# Build winner-by-metric table
# ------------------------------------------------------------

observation_rows = []

for metric in metric_names:

    if metric in best_models:

        observation_rows.append(
            {
                "Metric": metric,
                "Best Model":
                    best_models[metric]["model"],
                "Best Score":
                    best_models[metric]["score"],
            }
        )


observation_df = pd.DataFrame(
    observation_rows
)


st.markdown(
    "#### Best-performing Model by Metric"
)

st.dataframe(
    observation_df.style.format(
        {
            "Best Score": "{:.4f}"
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# MODEL-WISE PERFORMANCE OBSERVATIONS
# Assignment requirement - 3 marks
# ============================================================

st.markdown(
    "#### Model-wise Performance Observations"
)

st.write(
    "The following observations summarise the performance "
    "of each machine-learning model on the selected "
    "Dry Bean dataset."
)


# ------------------------------------------------------------
# Determine winner using weighted F1
# ------------------------------------------------------------

valid_f1_for_observations = (
    performance_df["F1"]
    .dropna()
)

overall_winner_for_observations = (
    valid_f1_for_observations.idxmax()
)


# ------------------------------------------------------------
# Function to create observation for each model
# ------------------------------------------------------------

def create_model_observation(
    current_model,
    row,
    overall_winner
):

    accuracy = row["Accuracy"]
    auc = row["AUC"]
    precision = row["Precision"]
    recall = row["Recall"]
    f1 = row["F1"]
    mcc = row["MCC"]

    if current_model == "Logistic Regression":

        explanation = (
            "The model achieved strong and balanced multiclass "
            "performance. The high AUC indicates effective "
            "separation among the bean classes using the "
            "linear classification approach."
        )

    elif current_model == "Decision Tree":

        explanation = (
            "The Decision Tree captured non-linear relationships "
            "in the dataset, although its overall test-set "
            "performance was below the strongest classifiers."
        )

    elif current_model == "KNN":

        explanation = (
            "KNN achieved competitive classification performance "
            "using neighbouring observations in the scaled "
            "feature space."
        )

    elif current_model == "Naive Bayes":

        explanation = (
            "Naive Bayes provides a probabilistic baseline. "
            "Its performance indicates that the model was less "
            "effective on this dataset than the other evaluated "
            "classifiers."
        )

    elif current_model in [
        "SVM",
        "Support Vector Machine",
    ]:

        explanation = (
            "The Support Vector Machine achieved strong and "
            "balanced multiclass performance using a non-linear "
            "decision boundary."
        )

    else:

        explanation = (
            "The model was evaluated using the same held-out "
            "test dataset as the other classifiers."
        )


    if current_model == overall_winner:

        explanation += (
            " This model achieved the highest weighted F1 "
            "score and is therefore selected as the overall winner."
        )


    return (
        f"Accuracy = {accuracy:.4f}, "
        f"AUC = {auc:.4f}, "
        f"Precision = {precision:.4f}, "
        f"Recall = {recall:.4f}, "
        f"F1 = {f1:.4f}, "
        f"MCC = {mcc:.4f}. "
        f"{explanation}"
    )


# ------------------------------------------------------------
# Generate model-wise observations
# ------------------------------------------------------------

model_observation_rows = []


for current_model, current_row in performance_df.iterrows():

    display_model_name = (
        "Support Vector Machine"
        if current_model == "SVM"
        else current_model
    )

    observation = create_model_observation(
        current_model,
        current_row,
        overall_winner_for_observations,
    )

    model_observation_rows.append(
        {
            "ML Model Name": display_model_name,
            "Observation about Model Performance":
                observation,
        }
    )


# ------------------------------------------------------------
# Add Overall Winner row
# ------------------------------------------------------------

winning_row = performance_df.loc[
    overall_winner_for_observations
]

winner_display_name = (
    "Support Vector Machine"
    if overall_winner_for_observations == "SVM"
    else overall_winner_for_observations
)


model_observation_rows.append(
    {
        "ML Model Name":
            "Overall Winner for the Dataset",

        "Observation about Model Performance":
            (
                f"{winner_display_name} is selected as the "
                f"overall winner because it achieved the "
                f"highest weighted F1 score of "
                f"{winning_row['F1']:.4f}. "
                f"The winning model achieved Accuracy "
                f"{winning_row['Accuracy']:.4f}, "
                f"AUC {winning_row['AUC']:.4f}, "
                f"Precision {winning_row['Precision']:.4f}, "
                f"Recall {winning_row['Recall']:.4f}, "
                f"and MCC {winning_row['MCC']:.4f}."
            ),
    }
)


# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

model_observation_df = pd.DataFrame(
    model_observation_rows
)


# ------------------------------------------------------------
# Display assignment-required table
# ------------------------------------------------------------

# ------------------------------------------------------------
# Display model-wise observations with word wrapping
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .model-observation-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .model-observation-table th {
        background-color: #f0f2f6;
        color: #31333f;
        font-weight: 600;
        text-align: left;
        padding: 12px;
        border: 1px solid #d9d9d9;
    }

    .model-observation-table td {
        padding: 12px;
        border: 1px solid #d9d9d9;
        vertical-align: top;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.5;
    }

    .model-observation-table th:first-child,
    .model-observation-table td:first-child {
        width: 22%;
        font-weight: 600;
    }

    .model-observation-table th:nth-child(2),
    .model-observation-table td:nth-child(2) {
        width: 78%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Convert the DataFrame to HTML
observation_html = model_observation_df.to_html(
    index=False,
    escape=True,
    classes="model-observation-table"
)

st.markdown(
    observation_html,
    unsafe_allow_html=True
)

# ============================================================
# 16. OVERALL MODEL WINNER
# ============================================================

st.markdown(
    "#### 🏆 Overall Winner"
)


# Weighted F1 is used as the primary selection criterion.
#
# F1 considers both Precision and Recall.
# Weighted averaging additionally accounts for the support
# of each class in the multiclass dataset.

valid_f1 = performance_df[
    "F1"
].dropna()


if not valid_f1.empty:

    overall_winner = valid_f1.idxmax()

    winner_metrics = performance_df.loc[
        overall_winner
    ]


    st.success(
        f"🏆 Overall Best Model: {overall_winner}"
    )


    # --------------------------------------------------------
    # Winner metric cards
    # --------------------------------------------------------

    winner_columns = st.columns(6)

    for column, metric in zip(
        winner_columns,
        metric_names,
    ):

        value = winner_metrics[
            metric
        ]

        if pd.notna(value):

            column.metric(
                metric,
                f"{value:.4f}",
            )

        else:

            column.metric(
                metric,
                "N/A",
            )


    # ========================================================
    # 17. AUTOMATIC OBSERVATIONS
    # ========================================================

    st.markdown(
        "#### Performance Observations"
    )


    if "Accuracy" in best_models:

        st.write(
            f"**1. Accuracy:** "
            f"{best_models['Accuracy']['model']} achieved "
            f"the highest classification accuracy of "
            f"**{best_models['Accuracy']['score']:.4f}**."
        )


    if "AUC" in best_models:

        st.write(
            f"**2. AUC:** "
            f"{best_models['AUC']['model']} achieved "
            f"the highest multiclass AUC of "
            f"**{best_models['AUC']['score']:.4f}**."
        )


    if "Precision" in best_models:

        st.write(
            f"**3. Precision:** "
            f"{best_models['Precision']['model']} achieved "
            f"the highest weighted Precision of "
            f"**{best_models['Precision']['score']:.4f}**."
        )


    if "Recall" in best_models:

        st.write(
            f"**4. Recall:** "
            f"{best_models['Recall']['model']} achieved "
            f"the highest weighted Recall of "
            f"**{best_models['Recall']['score']:.4f}**."
        )


    if "F1" in best_models:

        st.write(
            f"**5. F1 Score:** "
            f"{best_models['F1']['model']} achieved "
            f"the highest weighted F1 score of "
            f"**{best_models['F1']['score']:.4f}**."
        )


    if "MCC" in best_models:

        st.write(
            f"**6. Matthews Correlation Coefficient:** "
            f"{best_models['MCC']['model']} achieved "
            f"the highest MCC of "
            f"**{best_models['MCC']['score']:.4f}**."
        )


    # ========================================================
    # 18. OVERALL CONCLUSION
    # ========================================================

    st.markdown(
        "#### Overall Performance Conclusion"
    )

    st.write(
        f"Based on the held-out test dataset, "
        f"**{overall_winner}** is selected as the overall "
        f"winner because it achieved the highest weighted "
        f"F1 score of **{winner_metrics['F1']:.4f}**."
    )

    st.write(
        f"The winning model achieved an Accuracy of "
        f"**{winner_metrics['Accuracy']:.4f}**, "
        f"AUC of **{winner_metrics['AUC']:.4f}**, "
        f"Precision of **{winner_metrics['Precision']:.4f}**, "
        f"Recall of **{winner_metrics['Recall']:.4f}**, "
        f"F1 Score of **{winner_metrics['F1']:.4f}**, "
        f"and MCC of **{winner_metrics['MCC']:.4f}**."
    )

    st.info(
        "Overall winner selection criterion: Weighted F1 Score. "
        "Weighted F1 combines precision and recall and weights "
        "each class according to its number of observations. "
        "Accuracy, AUC and MCC are retained as supporting "
        "measures for the final model comparison."
    )


else:

    st.warning(
        "F1 scores are unavailable, so an overall winner "
        "cannot be determined."
    )


# ============================================================
# 19. TEST DATA UPLOAD
# ============================================================

st.subheader(
    "5. Evaluate Using Test Data"
)

upload = st.file_uploader(
    (
        f"Upload test data CSV "
        f"(include the '{TARGET_COLUMN}' column "
        f"for evaluation)"
    ),
    type="csv",
)


# ============================================================
# 20. PROCESS UPLOADED FILE
# ============================================================

if upload is not None:

    try:

        frame = pd.read_csv(
            upload
        )

    except Exception as error:

        st.error(
            f"Unable to read uploaded CSV: {error}"
        )

        st.stop()


    st.write(
        f"Uploaded records: **{len(frame):,}**"
    )


    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in frame.columns
    ]


    if missing_features:

        st.error(
            "Missing feature columns: "
            + ", ".join(
                sorted(missing_features)
            )
        )

        st.stop()


    # Preserve training feature order

    X = frame[
        FEATURE_NAMES
    ].copy()


    # ========================================================
    # 21. TEST DATA WITH CLASS COLUMN
    # ========================================================

    if TARGET_COLUMN in frame.columns:

        actual_labels = (
            frame[TARGET_COLUMN]
            .astype(str)
        )


        # ----------------------------------------------------
        # Validate class values
        # ----------------------------------------------------

        unknown_classes = sorted(
            set(actual_labels)
            - set(CLASS_NAMES)
        )


        if unknown_classes:

            st.error(
                "Unknown bean classes found in the "
                "uploaded CSV: "
                + ", ".join(
                    unknown_classes
                )
            )

            st.write(
                "Expected class values:"
            )

            st.write(
                ", ".join(CLASS_NAMES)
            )

            st.stop()


        # ----------------------------------------------------
        # Encode actual class labels
        # ----------------------------------------------------

        y = label_encoder.transform(
            actual_labels
        )


        # ----------------------------------------------------
        # Evaluate selected model
        # ----------------------------------------------------

        try:

            live_metrics, pred_encoded = score(
                model,
                X,
                y,
            )

        except Exception as error:

            st.error(
                f"Model evaluation failed: {error}"
            )

            st.stop()


        # ----------------------------------------------------
        # Decode predictions
        # ----------------------------------------------------

        pred_labels = (
            label_encoder.inverse_transform(
                pred_encoded
            )
        )


        # ====================================================
        # 22. UPLOADED DATA EVALUATION
        # ====================================================

        st.subheader(
            "Uploaded Test-data Evaluation"
        )


        live_columns = st.columns(6)


        for column, metric in zip(
            live_columns,
            metric_names,
        ):

            column.metric(
                metric,
                f"{live_metrics[metric]:.4f}",
            )


        # ====================================================
        # 23. CONFUSION MATRIX
        # ====================================================

        st.subheader(
            "Confusion Matrix"
        )


        cm = confusion_matrix(
            actual_labels,
            pred_labels,
            labels=CLASS_NAMES,
        )


        fig, ax = plt.subplots(
            figsize=(10, 7)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )


        ax.set_xlabel(
            "Predicted Class"
        )

        ax.set_ylabel(
            "Actual Class"
        )

        ax.set_title(
            f"Confusion Matrix - {model_name}"
        )


        plt.xticks(
            rotation=45,
            ha="right",
        )

        plt.yticks(
            rotation=0,
        )

        plt.tight_layout()


        st.pyplot(
            fig
        )

        plt.close(fig)


        # ====================================================
        # 24. CLASSIFICATION REPORT
        # ====================================================

        st.subheader(
            "Classification Report"
        )


        report_data = classification_report(
            actual_labels,
            pred_labels,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )


        accuracy_value = accuracy_score(
            actual_labels,
            pred_labels,
        )


        report = pd.DataFrame(
            report_data
        ).transpose()


        # Add accuracy row

        report.loc[
            "accuracy"
        ] = {

            "precision": accuracy_value,
            "recall": accuracy_value,
            "f1-score": accuracy_value,
            "support": len(frame),
        }


        # Add micro average

        report.loc[
            "micro avg"
        ] = {

            "precision": precision_score(
                actual_labels,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "recall": recall_score(
                actual_labels,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "f1-score": f1_score(
                actual_labels,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "support": len(frame),
        }


        # ----------------------------------------------------
        # Classification report ordering
        # ----------------------------------------------------

        row_order = (
            CLASS_NAMES
            + [
                "accuracy",
                "micro avg",
                "macro avg",
                "weighted avg",
            ]
        )


        available_rows = [
            row
            for row in row_order
            if row in report.index
        ]


        report = report.loc[
            available_rows
        ]


        # ====================================================
        # 25. CLASS-WISE SCORES
        # ====================================================

        st.markdown(
            "#### 1. Scores for the Seven Bean Varieties"
        )


        variety_rows = [
            class_name
            for class_name in CLASS_NAMES
            if class_name in report.index
        ]


        variety_report = report.loc[
            variety_rows
        ]

        variety_table_height = (
            35 * (len(variety_report) + 1)
        )
        st.dataframe(
            style_report_table(
                variety_report,
                "Bean Variety",
            ),
            use_container_width=True,
            height=variety_table_height,
        )


        # ====================================================
        # 26. ACCURACY AND MCC
        # ====================================================

        st.markdown(
            "#### 2. Overall Accuracy and MCC"
        )


        accuracy_column, mcc_column, (
            support_column
        ) = st.columns(3)


        accuracy_column.metric(
            "Accuracy",
            f"{accuracy_value:.4f}",
        )


        mcc_column.metric(
            "Matthews Correlation Coefficient (MCC)",
            f"{live_metrics['MCC']:.4f}",
        )


        support_column.metric(
            "Test Records",
            f"{len(frame):,}",
        )


        # ====================================================
        # 27. AVERAGE PERFORMANCE
        # ====================================================

        st.markdown(
            "#### 3. Average Performance Scores"
        )


        average_rows = [
            row
            for row in [
                "micro avg",
                "macro avg",
                "weighted avg",
            ]
            if row in report.index
        ]


        average_report = report.loc[
            average_rows
        ]

        average_table_height = (
            35 * (len(average_report) + 1)
        )
        st.dataframe(
            style_report_table(
                average_report,
                "Average Type",
            ),
            use_container_width=True,
            height=average_table_height,
        )


        # ====================================================
        # 28. PREDICTION RESULTS
        # ====================================================

        st.markdown(
            "#### 4. Prediction Results"
        )


        prediction_results = (
            frame.copy()
        )


        prediction_results[
            "Predicted_Class"
        ] = pred_labels


        prediction_results[
            "Correct_Prediction"
        ] = (
            actual_labels.values
            == pred_labels
        )


        # ----------------------------------------------------
        # Prediction confidence
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(X)
            )


            prediction_results[
                "Prediction_Confidence"
            ] = probabilities.max(
                axis=1
            )


        st.dataframe(
            prediction_results.head(50),
            use_container_width=True,
        )


        # ====================================================
        # 29. DOWNLOAD RESULTS
        # ====================================================

        csv_output = (
            prediction_results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )


        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_output,
            file_name="dry_bean_predictions.csv",
            mime="text/csv",
            key="download_evaluated_predictions",
            on_click="ignore",
        )


    # ========================================================
    # 30. TEST DATA WITHOUT CLASS COLUMN
    # ========================================================

    else:

        st.info(
            f"The uploaded CSV does not contain the "
            f"'{TARGET_COLUMN}' column. Predictions can be "
            f"generated, but evaluation metrics cannot "
            f"be calculated."
        )


        try:

            pred_encoded = model.predict(
                X
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()


        pred_labels = (
            label_encoder.inverse_transform(
                pred_encoded
            )
        )


        prediction_results = (
            frame.copy()
        )


        prediction_results[
            "Predicted_Class"
        ] = pred_labels


        # ----------------------------------------------------
        # Prediction confidence
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(X)
            )


            prediction_results[
                "Prediction_Confidence"
            ] = probabilities.max(
                axis=1
            )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        st.subheader(
            "Prediction Results"
        )


        st.dataframe(
            prediction_results.head(50),
            use_container_width=True,
        )


        # ----------------------------------------------------
        # Download predictions
        # ----------------------------------------------------

        csv_output = (
            prediction_results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )


        st.download_button(
            label="⬇️ Download Predictions",
            data=csv_output,
            file_name="dry_bean_predictions.csv",
            mime="text/csv",
            key="download_prediction_only_results",
            on_click="ignore",
        )


# ============================================================
# 31. NO TEST FILE UPLOADED
# ============================================================

else:

    st.info(
        "Upload test_data.csv to calculate evaluation metrics, "
        "the confusion matrix, classification report and "
        "individual prediction results."
    )


# ============================================================
# 32. FOOTER
# ============================================================

st.divider()

st.caption(
    "UCI Dry Bean comparative multiclass classification study using | "
    "Logistic Regression | Decision Tree | "
    "KNN | Naive Bayes | Support Vector Machine"
)