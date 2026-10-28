# UCI Dry Bean Classification - Streamlit Application

## a. Problem Statement

The objective of this project is to develop and compare five machine-learning classifiers that identify a dry-bean variety from its measured geometric properties.

This is a seven-class classification problem. The Streamlit application supports:

- Selection of any one of the five trained classification models
- Uploading test data in CSV format
- Displaying held-out test-set evaluation metrics
- Comparing the performance of all five models
- Identifying the best model for each evaluation metric
- Selecting an overall winning model
- Generating predictions for new data
- Displaying a confusion matrix
- Displaying a detailed classification report
- Downloading prediction results

---

## b. Dataset Description

**Dataset:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

The dataset contains geometric measurements extracted from images of dry-bean grains. The original Excel dataset is stored under the `data/` directory so that model training can be reproduced.

| Item | Description |
|---|---|
| Classification type | Multiclass classification |
| Input features | 16 numerical features |
| Target column | `Class` |
| Number of classes | 7 bean varieties |
| Classes | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA |
| Feature examples | Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, Solidity and ShapeFactor1-4 |
| Train/test method | Stratified train/test split |
| Random state | 42 |

Dataset counts, preprocessing details, train/test record counts and model results are updated automatically in the generated-results section below.

---

## c. GitHub Repository Link

**Repository URL:** `https://github.com/<your-github-username>/uci-dry-bean-streamlit-classification`

---

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

### Overall Winner

**SVM** is selected as the overall
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

---

## d. Models Used

The project trains and evaluates the following five machine-learning classification models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

The same training and held-out testing partitions are used for all five models to ensure a consistent comparison.

Standardisation is included in the pipelines for models that are sensitive to feature scale.

---

## e. Evaluation Metrics

The following metrics are used to evaluate the classification models:

### Accuracy

Accuracy measures the proportion of test records that are classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### AUC

Multiclass AUC measures the model's ability to distinguish between the seven bean classes. The application uses a one-vs-rest multiclass calculation.

### Precision

Precision measures how many records predicted as a particular class actually belong to that class.

```text
Precision = True Positives / (True Positives + False Positives)
```

### Recall

Recall measures how many records belonging to a particular class are identified correctly.

```text
Recall = True Positives / (True Positives + False Negatives)
```

### F1 Score

F1 is the harmonic mean of Precision and Recall.

```text
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

### Matthews Correlation Coefficient

MCC measures the correlation between actual and predicted classes. It provides an additional balanced assessment of classification quality.

### Overall Winner Selection

The overall winner is selected dynamically using the highest weighted F1 score.

Weighted F1 is used as the primary model-selection measure because it combines Precision and Recall while considering the number of observations belonging to each class.

Accuracy, AUC and MCC are retained as supporting measures for the final comparison.

---

## f. Data Preparation and Train/Test Split

The model-training process follows these stages:

```text
Original Dry Bean Dataset
          |
          v
Column-name standardisation
          |
          v
Duplicate-record detection and removal
          |
          v
Missing-value validation and removal
          |
          v
Model-ready dataset
          |
          v
Stratified train/test split
        /   \
       /     \
      v       v
Training     Testing
Dataset      Dataset
```

The training and testing datasets are written to:

```text
data/training_data.csv
test_data.csv
```

When `train_models.py` is executed again, the existing CSV contents are overwritten with the newly generated split.

The split uses:

```python
test_size=0.20
random_state=42
stratify=y
```

Using a fixed random state makes the split reproducible when the same source file and preprocessing logic are used.

---

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

### Main Files

- `app.py`: Streamlit application
- `train_models.py`: Data preparation, model training, evaluation and artefact generation
- `README.md`: Project documentation with a protected automatic-results section
- `requirements.txt`: Python package dependencies
- `test_data.csv`: Held-out testing dataset
- `data/training_data.csv`: Training dataset
- `model/metrics.json`: Evaluation results for all five models
- `model/metadata.json`: Dataset, split, class and model-file metadata
- `model/label_encoder.joblib`: Encoder used to convert between class names and numerical model labels
- `model/*.joblib`: Trained model pipelines

---

## h. Generated Files

Running `train_models.py` regenerates or updates:

- `data/training_data.csv`
- `test_data.csv`
- Five trained model files
- `model/label_encoder.joblib`
- `model/metrics.json`
- `model/metadata.json`
- The protected automatic-results section in `README.md`

Existing training and testing CSV contents are replaced rather than appended.

Manual content outside the automatically generated README markers is preserved.

---

## i. Installation

Create or activate the required Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The project requires packages such as:

- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib
- matplotlib
- seaborn
- openpyxl

The installable package name is `scikit-learn`, not `sklearn`.

---

## j. Model Training

Execute:

```bash
python train_models.py
```

The training script will:

1. Load the source Excel dataset
2. Validate and clean the data
3. Detect and remove duplicate records
4. Separate the input features and target
5. Encode the class labels
6. Create the stratified training and testing partitions
7. Overwrite the existing training and test CSV files
8. Train the five classification models
9. Evaluate each model on the held-out test set
10. Save the trained models
11. Save the label encoder
12. Update `metrics.json`
13. Update `metadata.json`
14. Update the protected results section in this README

---

## k. Running the Streamlit Application

Execute:

```bash
python -m streamlit run app.py
```

The application loads the trained models and supporting artefacts from the `model/` directory.

The application requires these generated files:

```text
model/
|-- logistic_regression.joblib
|-- decision_tree.joblib
|-- knn.joblib
|-- naive_bayes.joblib
|-- svm.joblib
|-- label_encoder.joblib
|-- metrics.json
`-- metadata.json
```

Run `train_models.py` before starting Streamlit if these files have not yet been generated.

---

## l. Streamlit Application Features

The application provides:

- Dataset preparation summary
- Original, duplicate and model-ready record counts
- Training and testing record counts
- Training and testing percentages
- Model-selection dropdown
- Held-out test-set performance metrics
- Five-model comparison table
- Best-performing model for every metric
- Dynamically selected overall winner
- CSV test-data upload
- Uploaded-data evaluation
- Seven-class confusion matrix
- Class-level classification report
- Micro, macro and weighted average scores
- Prediction confidence
- Downloadable prediction results

Use `test_data.csv` as the upload template. The file contains the required feature columns and the `Class` target column.

If an uploaded CSV does not contain the `Class` column, the application generates predictions but does not calculate evaluation metrics.

---

## m. Deploying to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the project files and generated model artefacts.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select **New app**.
5. Select the repository and branch.
6. Set the main application file to:

```text
app.py
```

7. Deploy the application.

Before deployment, confirm that GitHub contains:

- `app.py`
- `requirements.txt`
- `model/metadata.json`
- `model/metrics.json`
- `model/label_encoder.joblib`
- All five saved models

---

## n. Updating Model Results

After changing the dataset, preprocessing logic or model configuration, execute:

```bash
python train_models.py
```

The latest dataset statistics and model results will be inserted between:

```text
AUTO-GENERATED-RESULTS:START
```

and:

```text
AUTO-GENERATED-RESULTS:END
```

Manual sections outside these markers remain unchanged.

After retraining, commit the updated artefacts:

```bash
git add README.md
git add data/training_data.csv
git add test_data.csv
git add model/metrics.json
git add model/metadata.json
git add model/*.joblib
git commit -m "Update trained models and evaluation results"
git push
```

---

## o. Notes

- Do not manually edit the generated-results section.
- Do not remove either automatic-results marker.
- Keep both markers in the correct order.
- Maintain assignment explanations outside the generated section.
- Retrain models before publishing new evaluation results.
- Ensure `metadata.json`, `metrics.json` and the saved models originate from the same training execution.
