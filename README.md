# Machine Learning Assignment: UCI Dry Bean Classification Using Five ML Models

## a. Problem Statement

The objective of this project is to develop, evaluate and compare five machine-learning classifiers for identifying dry-bean varieties from their measured geometric properties.

This is a seven-class multiclass classification problem.

The project evaluates the following five machine-learning models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

The Streamlit application supports:

- Selection of any one of the five trained classification models
- Display of held-out test-set evaluation metrics
- Comparison of all five machine-learning models
- Best-performing model identification for each evaluation metric
- Model-wise performance observations for all five classifiers
- Dynamic identification of the overall winning model
- Uploading test data in CSV format
- Prediction of dry-bean varieties
- Confusion-matrix visualisation
- Detailed classification reports
- Micro, macro and weighted performance summaries
- Prediction-confidence display
- Downloading prediction results

## b. Dataset Description

**Dataset:** UCI Dry Bean Dataset

The dataset contains geometric measurements extracted from images of dry-bean grains. The original Excel dataset is stored in the `data/` directory so that model training can be reproduced.

The exact dataset counts, preprocessing results, train/test split statistics and model evaluation results are generated dynamically by `train_models.py` and written to the protected results section below.

## c. GitHub Repository Link

**Repository URL:** `https://github.com/99701a0554/ml_assignment_2_dry_bean_classification`

<!-- AUTO-GENERATED-RESULTS:START -->

## Latest Generated Results

> This section is generated automatically by
> `train_models.py`. Manual content outside these
> markers is preserved.

### Dataset Summary


| Dataset Measure | Value |
|---|---:|
| Original records | 13,611 |
| Duplicate records removed | 68 |
| Model-ready records | 13,543 |
| Training records | 10,834 (80.0%) |
| Test records | 2,709 (20.0%) |
| Numerical features | 16 |
| Bean classes | 7 |


The original dataset contained **13,611**
records. After preprocessing,
**13,543** records were available
for modelling.

The model-ready dataset was divided into:

- **10,834 training records**
  (80.0%)
- **2,709 test records**
  (20.0%)

### Bean Classes

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Model Performance Comparison

The models were evaluated on the same held-out
test dataset.

Weighted Precision, Recall and F1 are used for
multiclass evaluation.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9884 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Best Model by Metric

| Metric | Best Model | Score |
|---|---|---:|
| Accuracy | SVM | 0.9210 |
| AUC | Logistic Regression | 0.9934 |
| Precision | SVM | 0.9214 |
| Recall | SVM | 0.9210 |
| F1 | SVM | 0.9211 |
| MCC | SVM | 0.9045 |

### Model Performance Observations

- **Accuracy:** SVM achieved the highest score of **0.9210**.
- **AUC:** Logistic Regression achieved the highest score of **0.9934**.
- **Weighted Precision:** SVM achieved the highest score of **0.9214**.
- **Weighted Recall:** SVM achieved the highest score of **0.9210**.
- **Weighted F1 Score:** SVM achieved the highest score of **0.9211**.
- **Matthews Correlation Coefficient (MCC):** SVM achieved the highest score of **0.9045**.

### Model-wise Performance Observations

The following observations are calculated dynamically from
the latest held-out test-set evaluation.

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Accuracy **0.9192**, AUC **0.9934**, Precision **0.9197**, Recall **0.9192**, Weighted F1 **0.9193**, and MCC **0.9023**. This model ranked second by weighted F1. Its F1 score was 0.0018 below the winning model. It achieved the highest score for AUC. |
| **Decision Tree** | Accuracy **0.8966**, AUC **0.9363**, Precision **0.8965**, Recall **0.8966**, Weighted F1 **0.8964**, and MCC **0.8750**. This model ranked 4 out of 5 models by weighted F1. Its F1 score was 0.0247 below the winning model. It did not achieve the highest score for any of the evaluated metrics. |
| **KNN** | Accuracy **0.9155**, AUC **0.9811**, Precision **0.9163**, Recall **0.9155**, Weighted F1 **0.9157**, and MCC **0.8978**. This model ranked third by weighted F1. Its F1 score was 0.0054 below the winning model. It did not achieve the highest score for any of the evaluated metrics. |
| **Naive Bayes** | Accuracy **0.7630**, AUC **0.9644**, Precision **0.7647**, Recall **0.7630**, Weighted F1 **0.7607**, and MCC **0.7143**. This model ranked 5 out of 5 models by weighted F1. Its F1 score was 0.1604 below the winning model. It did not achieve the highest score for any of the evaluated metrics. |
| **Support Vector Machine** | Accuracy **0.9210**, AUC **0.9884**, Precision **0.9214**, Recall **0.9210**, Weighted F1 **0.9211**, and MCC **0.9045**. This model ranked first by weighted F1 and is selected as the overall winner. It achieved the highest score for Accuracy, Precision, Recall, Weighted F1, MCC. |
| **Overall Winner for the Dataset** | **Support Vector Machine** is selected as the overall winner because it achieved the highest weighted F1 score of **0.9211**. The model also achieved the highest score for Accuracy, Precision, Recall, Weighted F1, MCC. |

### Overall Winner

**Support Vector Machine** is selected as the overall
winning model based on the highest weighted F1
score of **0.9211**.

The overall winner achieved:

- Accuracy: **0.9210**
- AUC: **0.9884**
- Precision: **0.9214**
- Recall: **0.9210**
- Weighted F1: **0.9211**
- MCC: **0.9045**

Weighted F1 is used as the primary selection
criterion because it combines Precision and Recall
while accounting for class support.

<!-- AUTO-GENERATED-RESULTS:END -->

## d. Models Used

The project trains and evaluates five classification algorithms:

1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbours (KNN)**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVM)**

All five classifiers are evaluated using the same held-out test partition to provide a consistent model comparison.

Feature scaling is included in the pipelines for algorithms that are sensitive to differences in feature magnitude.

## e. Evaluation Metrics

The following evaluation metrics are used to compare the five classification models.

### Accuracy

Accuracy measures the proportion of test observations classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### AUC

Area Under the ROC Curve (AUC) measures a model's ability to distinguish between classes.

For this multiclass classification problem, AUC is calculated using a one-vs-rest approach.

### Precision

Precision measures the proportion of observations predicted as a class that actually belong to that class.

```text
Precision = True Positives / (True Positives + False Positives)
```

### Recall

Recall measures the proportion of observations belonging to a class that are correctly identified.

```text
Recall = True Positives / (True Positives + False Negatives)
```

### F1 Score

The F1 score is the harmonic mean of Precision and Recall.

```text
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

### Matthews Correlation Coefficient (MCC)

MCC measures the correlation between actual and predicted classifications and provides an additional balanced measure of classification quality.

### Weighted Multiclass Metrics

Weighted Precision, Recall and F1 are used for the primary multiclass model comparison.

The weighting considers the support, or number of observations, belonging to each class.

### Overall Model Selection

The overall winning model is selected dynamically from the latest model-evaluation results using weighted F1 as the primary selection criterion.

Accuracy, AUC, Precision, Recall and MCC are retained as supporting measures.

All calculated metric values, rankings and winner information are generated by `train_models.py`.

## f. Data Preparation and Train/Test Split

The model-training pipeline performs the following stages:

```text
Original UCI Dry Bean Dataset
            |
            v
      Load source data
            |
            v
Standardise column names
            |
            v
 Detect duplicate records
            |
            v
    Remove duplicates
            |
            v
Validate missing values
            |
            v
Remove incomplete records
      when applicable
            |
            v
   Model-ready dataset
            |
            v
Separate Features (X)
and Target Class (y)
            |
            v
   Encode class labels
            |
            v
Stratified train/test split
          /   \
         /     \
        v       v
   Training    Testing
    Dataset    Dataset
```

The generated training and testing datasets are stored as:

```text
data/training_data.csv
test_data.csv
```

When the training program is executed again, the existing CSV files are overwritten using the latest processed dataset and train/test split.

The exact original dataset count, duplicate count, model-ready record count, training/test counts and split percentages are generated dynamically by `train_models.py` and displayed in the automatically generated results section.

## g. Project Structure

```text
dry_bean_streamlit_project/
|
|-- app.py
|-- train_models.py
|-- README.md
|-- requirements.txt
|-- test_data.csv
|
|-- data/
|   |-- Dry_Bean_Dataset.xlsx
|   `-- training_data.csv
|
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
    |-- label_encoder.joblib
    |-- metrics.json
    `-- metadata.json
```

### Main Project Files

- `app.py` - Streamlit machine-learning application
- `train_models.py` - Data preprocessing, model training, evaluation and artefact generation
- `README.md` - Project documentation
- `requirements.txt` - Python package dependencies
- `data/Dry_Bean_Dataset.xlsx` - Source dataset
- `data/training_data.csv` - Generated training partition
- `test_data.csv` - Generated held-out test partition
- `model/metrics.json` - Latest evaluation results
- `model/metadata.json` - Dataset, split, model and class metadata
- `model/label_encoder.joblib` - Class-label encoder
- `model/*.joblib` - Saved trained model pipelines

## h. Generated Files

Running `train_models.py` regenerates or updates the machine-learning artefacts required by the project.

The generated outputs include:

- `data/training_data.csv` - Training partition generated from the processed dataset
- `test_data.csv` - Held-out testing partition
- Five trained model files in the `model/` directory
- `model/label_encoder.joblib` - Encoded mapping between bean class names and numerical labels
- `model/metrics.json` - Latest held-out test-set evaluation results
- `model/metadata.json` - Dataset, preprocessing, train/test split, feature, class and model-file metadata
- The protected automatic-results section in `README.md`

Existing training and testing CSV files are overwritten when model training is executed again rather than appended.

Calculated dataset counts, split percentages, evaluation metrics, model rankings, model-wise observations and the overall winner are generated from the current execution of `train_models.py` and are not manually maintained in this README.

Manual README content outside the automatic results markers is preserved.

## i. Installation

Install the required Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses packages including:

- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib
- matplotlib
- seaborn
- openpyxl

## j. Model Training

Run:

```bash
python train_models.py
```

The training program performs the complete machine-learning workflow:

1. Loads the source Dry Bean dataset
2. Validates and preprocesses the data
3. Detects and removes duplicate records
4. Validates missing data
5. Separates the input features and target
6. Encodes the target classes
7. Creates the stratified training and testing partitions
8. Updates the training and test CSV files
9. Trains all five classification models
10. Evaluates all models on the held-out test dataset
11. Saves the trained model pipelines
12. Saves the label encoder
13. Updates `metrics.json`
14. Updates `metadata.json`
15. Calculates the best model for each metric
16. Calculates model-wise performance observations
17. Selects the overall winner
18. Updates the protected results section in this README

All calculated dataset counts, performance statistics, rankings and winner information are generated from the current training execution.

## k. Running the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

The application loads the trained model artefacts and supporting metadata from the `model/` directory.

If the model artefacts have not been created or are outdated, run `train_models.py` before launching the Streamlit application.

## l. Streamlit Application Features

The Streamlit application provides:

- Dataset preparation summary
- Original dataset record count
- Duplicate-record count
- Model-ready record count
- Training and testing record counts
- Training and testing percentages
- Model-selection dropdown
- Held-out test-set evaluation
- Comparison of all five classifiers
- Best model by evaluation metric
- Model-wise performance observations for all five classifiers
- Dynamic overall-winner selection
- CSV test-data upload
- Uploaded-data evaluation
- Confusion matrix
- Per-class classification report
- Micro, macro and weighted average performance
- Prediction confidence
- Downloadable prediction results

Calculated results displayed by the application are obtained from the model-training artefacts rather than being manually entered.

## m. Deploying to Streamlit Community Cloud

Live Streamlit Application
Open the Live UCI Dry Bean Classification Application `https://uci-dry-bean-app-classification-ernopnau2nwqv7yabsspad.streamlit.app/`

The application is deployed using Streamlit Community Cloud and provides the interactive frontend for model comparison, test-data evaluation and prediction.

1. Push the project files and generated model artefacts to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application from the GitHub repository.
4. Select the appropriate repository and branch.
5. Set the main application file to `app.py`.
6. Deploy the application.

Before deployment, verify that the repository contains the latest:

- `app.py`
- `requirements.txt`
- `README.md`
- `model/metadata.json`
- `model/metrics.json`
- `model/label_encoder.joblib`
- All five saved model files

## n. Updating Model Results

Whenever the dataset, preprocessing logic, train/test configuration or model configuration changes, execute:

```bash
python train_models.py
```

The training program automatically recalculates and updates:

- Dataset statistics
- Data-cleaning statistics
- Training and testing counts
- Training and testing percentages
- Model-evaluation metrics
- Best model by metric
- Model rankings
- Model-wise performance observations
- Overall winner
- The protected results section of `README.md`

Manual README content outside the automatic markers is preserved.

After successful training, commit the refreshed artefacts to the Git repository.

## o. Notes

- Do not manually edit calculated results in the protected README section.
- Do not remove the automatic-results START or END markers.
- Keep assignment explanations outside the generated-results section.
- Run `train_models.py` whenever the dataset or model configuration changes.
- Ensure `metadata.json`, `metrics.json`, saved model files, generated CSV files and README results originate from the same training execution.
