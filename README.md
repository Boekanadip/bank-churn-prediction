# Bank Customer Churn Prediction

Machine Learning system for predicting the probability of customer churn based on customer demographic, financial, and account activity information.

The project covers the end-to-end Data Science workflow, from data preprocessing and exploratory data analysis (EDA), model training and evaluation, to model serving through a REST API using FastAPI.

---

## Project Overview

Customer churn prediction can help identify customers who have a higher probability of leaving a bank.

The objective of this project is to build a classification model that predicts:

* Whether a customer is likely to churn
* The probability of customer churn
* A risk level based on the predicted churn probability

The trained model is exposed through a FastAPI endpoint so that the prediction service can be consumed by other applications, such as a backend application or customer management system.

---

## Project Workflow

```text
Raw Dataset
    │
    ▼
Data Understanding & EDA
    │
    ▼
Data Cleaning & Preprocessing
    │
    ├── Missing Value Handling
    ├── Numerical Feature Scaling
    └── Categorical Feature Encoding
    │
    ▼
Train-Test Split
    │
    ▼
Model Training
    │
    ├── Logistic Regression
    ├── Random Forest
    └── XGBoost
    │
    ▼
Model Evaluation
    │
    ▼
Best Model Selection
    │
    ▼
model.pkl + preprocessor.pkl
    │
    ▼
FastAPI
    │
    ▼
POST /predict
    │
    ▼
Churn Probability & Risk Level
```

---

## Dataset

The project uses the **Bank Customer Churn Prediction** dataset.

### Main Features

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| `customer_id`      | Unique customer identifier                |
| `credit_score`     | Customer credit score                     |
| `country`          | Customer country                          |
| `gender`           | Customer gender                           |
| `age`              | Customer age                              |
| `tenure`           | Number of years as a bank customer        |
| `balance`          | Customer account balance                  |
| `products_number`  | Number of bank products used              |
| `credit_card`      | Whether the customer has a credit card    |
| `active_member`    | Whether the customer is an active member  |
| `estimated_salary` | Estimated customer salary                 |
| `churn`            | Target variable indicating customer churn |

`customer_id` is used only as an identifier and is not used as a model feature.

---

## Data Preprocessing

The preprocessing pipeline is implemented using Scikit-learn `Pipeline` and `ColumnTransformer`.

### Numerical Features

The following numerical features are processed using:

1. Median imputation
2. Standard scaling

```text
credit_score
age
tenure
balance
products_number
estimated_salary
```

### Categorical Features

The following categorical features are processed using:

1. Most-frequent imputation
2. One-hot encoding

```text
country
gender
credit_card
active_member
```

The preprocessing pipeline is saved as:

```text
preprocessor.pkl
```

This pipeline must be reused during model serving to ensure that incoming API data receives the same preprocessing treatment as the training data.

---

## Machine Learning

Several classification models are trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

Model evaluation considers multiple classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Because the target variable is imbalanced, model selection should not rely solely on accuracy.

The final production model is stored as:

```text
model.pkl
```

> The current `model.pkl` in this repository is an XGBoost classifier. The API model version should therefore reflect the actual production model.

---

## Model Serving

The trained model is served using **FastAPI**.

The API performs the following process:

```text
Client Application
      │
      ▼
POST /predict
      │
      ▼
Request Validation
      │
      ▼
Preprocessing
      │
      ▼
XGBoost Model
      │
      ▼
Churn Probability
      │
      ▼
Risk Classification
      │
      ▼
JSON Response
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

This endpoint can be used by the backend or DevOps team to verify that the ML service is running and the model has been loaded successfully.

---

### Churn Prediction

```http
POST /predict
```

Example request:

```json
{
  "customer_id": "a1b2c3d4-e5f6-47a8-9b12-cd34ef567890",
  "credit_score": 650,
  "country": "France",
  "gender": "Female",
  "age": 42,
  "tenure": 5,
  "balance": 125000.50,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 1,
  "estimated_salary": 78000.00
}
```

Example response:

```json
{
  "customer_id": "a1b2c3d4-e5f6-47a8-9b12-cd34ef567890",
  "churn_probability": 0.7345,
  "churn_percentage": 73,
  "risk_level": "Merah",
  "model_version": "v1.0-xgboost"
}
```

---

## Risk Level

The API converts the predicted churn probability into three risk levels:

| Probability     | Risk Level |
| --------------- | ---------- |
| `< 0.30`        | Hijau      |
| `0.30 – < 0.70` | Kuning     |
| `>= 0.70`       | Merah      |

These thresholds are currently implemented in the ML API layer so that the business logic remains consistent across consuming applications.

---

## Project Structure

```text
bank-churn-prediction/
│
├── api/
│   └── main.py
│
├── data/
│   └── Bank Customer Churn Prediction.csv
│
├── notebooks/
│   ├── Bank_Churn_DS_Workflow.ipynb
│   └── eda_bank_churn.py
│
├── src/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── preprocessing.py
│   └── train.py
│
├── README.md
├── requirements.txt
├── SETUP_ENVIRONMENT_ML.md
└── LESSON_LEARNED_TEMPLATE.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Boekanadip/bank-churn-prediction.git
cd bank-churn-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API

Before starting the API, make sure the following model artifacts are available in the location expected by `api/main.py`:

```text
model.pkl
preprocessor.pkl
```

Run FastAPI using Uvicorn:

```bash
uvicorn api.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Model Artifacts

The model serving system requires two artifacts:

### `model.pkl`

The trained machine learning model used to generate churn predictions.

### `preprocessor.pkl`

The fitted Scikit-learn preprocessing pipeline used to transform incoming customer data before prediction.

Both artifacts must be generated from the same training pipeline to prevent preprocessing mismatch between training and production.

---

## Data Science Contribution

The Data Science workflow in this project includes:

* Data understanding
* Data cleaning
* Exploratory Data Analysis
* Feature preparation
* Preprocessing pipeline development
* Model training
* Model comparison
* Model evaluation
* Model selection
* Model serialization
* ML API development
* API integration support for software development

The final model is prepared as a deployable artifact and exposed through FastAPI for integration with other software systems.

---

## Software Integration

The ML service is designed to work as a separate prediction service.

```text
                    ┌─────────────────────┐
                    │   Client / Backend   │
                    │   Application        │
                    └──────────┬──────────┘
                               │
                         HTTP Request
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │    ML Service       │
                    └──────────┬──────────┘
                               │
                     Preprocessing
                               │
                               ▼
                    ┌─────────────────────┐
                    │   XGBoost Model     │
                    │     model.pkl       │
                    └──────────┬──────────┘
                               │
                     Prediction Result
                               │
                               ▼
                    ┌─────────────────────┐
                    │   JSON Response     │
                    │ Probability + Risk  │
                    └─────────────────────┘
```

This separation allows the software application and machine learning model to be developed and maintained independently.

---

## Important Notes

* `customer_id` is not used as a model feature.
* The same preprocessing pipeline must be used during training and inference.
* `model.pkl` and `preprocessor.pkl` should be treated as a matching pair.
* Model version information in the API should match the actual serialized model.
* Changes to model features or preprocessing logic require retraining and regeneration of the model artifacts.
* The dataset used in this repository should not be replaced with confidential production customer data.
