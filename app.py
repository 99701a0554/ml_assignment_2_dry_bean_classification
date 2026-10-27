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
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

METRICS_FILE = MODEL_DIR / "metrics.json"
METADATA_FILE = MODEL_DIR / "metadata.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"


# ============================================================
# 2. DEFAULT MODEL FILES
# ============================================================

# These are used as a fallback if model_files is not available
# in metadata.json.

DEFAULT_MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="wide",
)


# ============================================================
# 4. LOAD APPLICATION ASSETS
# ============================================================

@st.cache_resource
def load_assets():
    """
    Load:
    1. metrics.json
    2. metadata.json
    3. label encoder
    4. five trained ML models
    """

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        METRICS_FILE,
        METADATA_FILE,
        LABEL_ENCODER_FILE,
    ]

    missing_files = [
        file
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing:\n"
            + "\n".join(str(file) for file in missing_files)
            + "\n\nRun train_models.py before starting Streamlit."
        )

    # --------------------------------------------------------
    # Load metrics
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # --------------------------------------------------------
    # Load label encoder
    # --------------------------------------------------------

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    # --------------------------------------------------------
    # Read model file names from metadata
    # --------------------------------------------------------

    model_files = metadata.get(
        "model_files",
        DEFAULT_MODEL_FILES,
    )

    # --------------------------------------------------------
    # Load all models
    # --------------------------------------------------------

    models = {}

    for model_name, file_name in model_files.items():

        model_path = MODEL_DIR / file_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
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
# 5. EVALUATION FUNCTION
# ============================================================

def score(model, X, y):
    """
    Evaluate the selected model on uploaded test data.

    IMPORTANT:
    y must contain encoded numerical class values.
    """

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
# 6. CLASSIFICATION REPORT STYLE
# ============================================================

def style_report_table(
    table,
    index_name,
):
    """
    Apply colour and numeric formatting to the
    classification report.
    """

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
                        (
                            "background-color",
                            "#0B3C5D",
                        ),
                        (
                            "color",
                            "white",
                        ),
                        (
                            "font-weight",
                            "bold",
                        ),
                        (
                            "text-align",
                            "center",
                        ),
                    ],
                },

                {
                    "selector": "td",
                    "props": [
                        (
                            "text-align",
                            "center",
                        ),
                        (
                            "border",
                            "1px solid #D6E4F0",
                        ),
                    ],
                },

                {
                    "selector":
                        "tbody tr:nth-child(even)",
                    "props": [
                        (
                            "background-color",
                            "#F4F9FD",
                        ),
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
        f"Unable to load application assets: {error}"
    )

    st.stop()


# ============================================================
# 8. READ THE NEW METADATA STRUCTURE
# ============================================================

FEATURE_NAMES = metadata["feature_names"]

CLASS_NAMES = metadata["class_names"]

TARGET_COLUMN = metadata["target_column"]

NUMBER_OF_FEATURES = metadata.get(
    "number_of_features",
    len(FEATURE_NAMES),
)

NUMBER_OF_CLASSES = metadata.get(
    "number_of_classes",
    len(CLASS_NAMES),
)


# ============================================================
# 9. APPLICATION HEADER
# ============================================================

st.title(
    "UCI Dry Bean Classification - Five ML Models"
)

st.caption(
    f"{NUMBER_OF_FEATURES} numerical features and "
    f"{NUMBER_OF_CLASSES} bean varieties. "
    "Weighted-average metrics are used for this "
    "multi-class classification application."
)


# ============================================================
# 10. DISPLAY DATASET INFORMATION
# ============================================================

with st.expander(
    "Dataset and Model Information"
):

    st.write(
        "**Target column:**",
        TARGET_COLUMN,
    )

    st.write(
        "**Number of features:**",
        NUMBER_OF_FEATURES,
    )

    st.write(
        "**Number of classes:**",
        NUMBER_OF_CLASSES,
    )

    st.write(
        "**Bean varieties:**"
    )

    for class_name in CLASS_NAMES:
        st.write(
            f"- {class_name}"
        )


# ============================================================
# 11. SELECT CLASSIFICATION MODEL
# ============================================================

st.subheader(
    "Select Classification Model"
)

model_name = st.selectbox(
    "Select classification model",
    list(models.keys()),
)

model = models[model_name]


# ============================================================
# 12. RETRIEVE STORED METRICS
# ============================================================

# IMPORTANT:
#
# OLD CODE:
# stored = metrics["models"][model_name]
#
# NEW CODE:
# metrics.json stores model names directly.

if model_name not in metrics:

    st.error(
        f"No stored metrics were found for "
        f"{model_name}."
    )

    st.stop()


stored = metrics[model_name]


# ============================================================
# 13. DISPLAY HELD-OUT TEST RESULTS
# ============================================================

st.subheader(
    "Held-out Test-set Evaluation"
)

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

metric_columns = st.columns(
    len(metric_names)
)

for column, metric in zip(
    metric_columns,
    metric_names,
):

    metric_value = stored.get(
        metric
    )

    if metric_value is None:

        column.metric(
            metric,
            "N/A",
        )

    else:

        column.metric(
            metric,
            f"{float(metric_value):.4f}",
        )


# ============================================================
# 14. MODEL COMPARISON TABLE
# ============================================================

st.subheader(
    "Comparison of All Five Models"
)

model_comparison = (
    pd.DataFrame(metrics)
    .T
    .reindex(columns=metric_names)
)

st.dataframe(
    model_comparison.style.format(
        "{:.4f}"
    ),
    use_container_width=True,
)


# ============================================================
# 15. FILE UPLOAD SECTION
# ============================================================

st.subheader(
    "Evaluate Using Test Data"
)

upload = st.file_uploader(
    (
        f"Upload test data CSV "
        f"(include the {TARGET_COLUMN} column "
        f"for evaluation)"
    ),
    type="csv",
)


# ============================================================
# 16. PROCESS UPLOADED CSV
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


    # --------------------------------------------------------
    # Display basic uploaded file information
    # --------------------------------------------------------

    st.write(
        f"Uploaded records: **{len(frame):,}**"
    )


    # --------------------------------------------------------
    # Check feature columns
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURE_NAMES
        if feature not in frame.columns
    ]


    if missing:

        st.error(
            "Missing feature columns: "
            + ", ".join(
                sorted(missing)
            )
        )

        st.stop()


    # --------------------------------------------------------
    # Select features in exactly the same order as training
    # --------------------------------------------------------

    X = frame[
        FEATURE_NAMES
    ].copy()


    # ========================================================
    # 17. CSV CONTAINS ACTUAL CLASS
    # ========================================================

    if TARGET_COLUMN in frame.columns:

        # ----------------------------------------------------
        # Validate class labels
        # ----------------------------------------------------

        uploaded_classes = (
            frame[TARGET_COLUMN]
            .astype(str)
        )

        unknown_classes = sorted(
            set(uploaded_classes)
            - set(CLASS_NAMES)
        )


        if unknown_classes:

            st.error(
                "The uploaded file contains unknown "
                "bean classes: "
                + ", ".join(
                    unknown_classes
                )
            )

            st.write(
                "Expected classes:",
                ", ".join(CLASS_NAMES),
            )

            st.stop()


        # ----------------------------------------------------
        # Encode actual string classes
        # ----------------------------------------------------

        y = label_encoder.transform(
            uploaded_classes
        )


        # ----------------------------------------------------
        # Evaluate model
        # ----------------------------------------------------

        live, pred_encoded = score(
            model,
            X,
            y,
        )


        # ----------------------------------------------------
        # Decode predictions
        # ----------------------------------------------------

        pred_labels = (
            label_encoder.inverse_transform(
                pred_encoded
            )
        )


        # ====================================================
        # 18. LIVE TEST-DATA EVALUATION
        # ====================================================

        st.subheader(
            "Uploaded Test-data Evaluation"
        )

        live_df = pd.DataFrame(
            [live]
        )

        st.dataframe(
            live_df.style.format(
                "{:.4f}"
            ),
            use_container_width=True,
        )


        # ====================================================
        # 19. CONFUSION MATRIX
        # ====================================================

        st.subheader(
            "Confusion Matrix"
        )

        # Work with readable class labels rather than
        # encoded numbers in the visualisation.

        cm = confusion_matrix(
            uploaded_classes,
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
        # 20. CLASSIFICATION REPORT
        # ====================================================

        st.subheader(
            "Classification Report"
        )

        report_data = classification_report(
            uploaded_classes,
            pred_labels,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy_value = accuracy_score(
            uploaded_classes,
            pred_labels,
        )


        # ----------------------------------------------------
        # Convert report to DataFrame
        # ----------------------------------------------------

        report = pd.DataFrame(
            report_data
        ).transpose()


        # ----------------------------------------------------
        # Add accuracy row
        # ----------------------------------------------------

        report.loc[
            "accuracy"
        ] = {
            "precision": accuracy_value,
            "recall": accuracy_value,
            "f1-score": accuracy_value,
            "support": len(frame),
        }


        # ----------------------------------------------------
        # Add micro-average row
        # ----------------------------------------------------

        report.loc[
            "micro avg"
        ] = {
            "precision": precision_score(
                uploaded_classes,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "recall": recall_score(
                uploaded_classes,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "f1-score": f1_score(
                uploaded_classes,
                pred_labels,
                average="micro",
                zero_division=0,
            ),

            "support": len(frame),
        }


        # ----------------------------------------------------
        # Arrange report
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
        # 21. BEAN VARIETY RESULTS
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

        st.dataframe(
            style_report_table(
                variety_report,
                "Bean Variety",
            ),
            use_container_width=True,
            height=330,
        )


        # ====================================================
        # 22. ACCURACY AND MCC
        # ====================================================

        st.markdown(
            "#### 2. Overall Accuracy and MCC"
        )

        accuracy_column, (
            mcc_column
        ), support_column = st.columns(3)


        accuracy_column.metric(
            "Accuracy",
            f"{accuracy_value:.4f}",
        )


        mcc_column.metric(
            "Matthews Correlation Coefficient (MCC)",
            f"{live['MCC']:.4f}",
        )


        support_column.metric(
            "Test Records (Support)",
            f"{len(frame):,}",
        )


        # ====================================================
        # 23. AVERAGE PERFORMANCE
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

        st.dataframe(
            style_report_table(
                average_report,
                "Average Type",
            ),
            use_container_width=True,
            height=190,
        )


        # ====================================================
        # 24. DISPLAY PREDICTIONS
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
            prediction_results[
                TARGET_COLUMN
            ].astype(str)
            == prediction_results[
                "Predicted_Class"
            ]
        )


        # ----------------------------------------------------
        # Probability / confidence
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
        # 25. DOWNLOAD RESULTS
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
            label="Download Prediction Results",
            data=csv_output,
            file_name=(
                "dry_bean_predictions.csv"
            ),
            mime="text/csv",
        )


    # ========================================================
    # 26. CSV DOES NOT CONTAIN CLASS
    # ========================================================

    else:

        st.info(
            f"No '{TARGET_COLUMN}' column was supplied. "
            "Predictions will be generated, but "
            "evaluation metrics cannot be calculated."
        )


        # ----------------------------------------------------
        # Generate encoded predictions
        # ----------------------------------------------------

        pred_encoded = model.predict(
            X
        )


        # ----------------------------------------------------
        # Convert encoded predictions back to bean names
        # ----------------------------------------------------

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
        # Add prediction confidence
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
        # Display predictions
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
            label="Download Predictions",
            data=csv_output,
            file_name=(
                "dry_bean_predictions.csv"
            ),
            mime="text/csv",
        )


# ============================================================
# 27. NO CSV UPLOADED
# ============================================================

else:

    st.info(
        "Upload test_data.csv to calculate metrics, "
        "a confusion matrix and a classification report."
    )

    # IMPORTANT:
    #
    # The new metrics.json contains scalar evaluation
    # metrics only. It does NOT contain:
    #
    # stored["confusion_matrix"]
    #
    # Therefore the confusion matrix is generated only
    # after test_data.csv has been uploaded.


# ============================================================
# 28. FOOTER
# ============================================================

st.divider()

st.caption(
    "Dry Bean Classification | "
    "Logistic Regression | Decision Tree | "
    "KNN | Naive Bayes | Support Vector Machine"
)