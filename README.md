# UCI Dry Bean Classification - Streamlit Application

## a. Problem statement

Develop and compare five machine-learning classifiers that identify a dry-bean variety from its measured geometric properties. This is a seven-class classification problem. The Streamlit application provides model selection, test-data upload, evaluation metrics, a confusion matrix, and a classification report.

## b. Dataset description

**Dataset:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), collected by Koklu and Ozkan (2020).

| Item | Description |
|---|---|
| Classification type | Multi-class: 7 bean varieties |
| Instances | 13,611 |
| Input features | 16 numerical features |
| Classes | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Feature examples | Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, Solidity and ShapeFactor1-4 |
| Missing values | None |
| Train/test split | 80% / 20%, stratified, `random_state=42` |
| Test instances | 2,723 |

The data consists of shape measurements extracted from high-resolution images of dry-bean grains. The original UCI Excel file is included in `data/` for reproducible retraining.

## c. GitHub repository link

**Repository URL:** `https://github.com/<your-github-username>/uci-dry-bean-streamlit-classification`

Replace `<your-github-username>` after uploading this folder to your GitHub account.

## d. Models used and evaluation metrics

All models were trained on the same 80% training partition and evaluated on the same 20% held-out test partition. AUC is macro-averaged one-vs-rest ROC-AUC; precision, recall, and F1 are macro averages.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9349 | 0.9305 | 0.9325 | 0.9050 |
| Decision Tree | 0.8953 | 0.9540 | 0.9104 | 0.9090 | 0.9094 | 0.8735 |
| K-Nearest Neighbors | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9243 | 0.9946 | 0.9376 | 0.9354 | 0.9365 | 0.9085 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong performance on scaled numerical features; it achieved the second-highest AUC and well-balanced macro metrics. |
| Decision Tree | Easy to explain but less accurate than the distance- and margin-based models; one tree can create boundaries that generalize less smoothly. |
| K-Nearest Neighbors | Performed very well after scaling because the geometric measurements produce useful local neighbourhoods. |
| Gaussian Naive Bayes | Fast but the weakest classifier, because the feature-independence assumption is unsuitable for correlated geometric measurements. |
| Support Vector Machine | Best overall result: highest accuracy, macro precision, recall, F1, and MCC. Its RBF kernel captures non-linear class boundaries. |

**Overall winner: Support Vector Machine.** It achieved the highest accuracy (0.9243), macro F1 (0.9365), and MCC (0.9085). Logistic Regression has a marginally higher AUC (0.9948), making it a good interpretable alternative.

## Project structure

```text
project-dry-bean/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/
|   |-- Dry_Bean_Dataset.xlsx
|   `-- training_data.csv
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    `-- metrics.json
```

## Run locally

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload this project folder.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), select **New app**, then select the repository and `main` branch.
3. Set the main file path to `app.py` and click **Deploy**.

## App features

- CSV test-data upload
- Dropdown for all five classifiers
- Accuracy, macro AUC, macro precision, macro recall, macro F1 and MCC
- Seven-class confusion matrix and classification report
- Prediction output when the uploaded CSV does not contain `Class`

Use the supplied `test_data.csv` as the upload template. It contains all 16 feature columns and the `Class` target column.
