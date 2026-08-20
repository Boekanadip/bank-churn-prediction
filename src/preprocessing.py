"""
src/preprocessing.py

Preprocessing pipeline — SATU-SATUNYA sumber kebenaran untuk tahap ini.
Dijalankan sekali sebelum src/train.py.

Cara jalanin (dari root folder project):
    python src/preprocessing.py

Output (semua tersimpan di folder src/, sejajar dengan file ini):
    - src/preprocessor.pkl        (pipeline, dipakai ulang di api/main.py)
    - src/X_train_processed.npy
    - src/X_test_processed.npy
    - src/y_train.csv
    - src/y_test.csv
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================================================
# Konfigurasi path — SEMUA relatif terhadap lokasi file ini,
# supaya tidak error kalau dijalankan dari folder berbeda.
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../bank-churn-prediction/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../bank-churn-prediction
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "churn.csv")
OUTPUT_DIR = BASE_DIR  # simpan output di src/, sejajar file ini

# =========================================================
# Definisi kolom — dipakai ulang di train.py lewat import
# =========================================================
ID_COL = "CustomerId"
TARGET_COL = "Exited"
NUMERICAL_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]

def build_preprocessor() -> ColumnTransformer:
    """Bangun ColumnTransformer baru (belum di-fit)."""
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    print("Membaca data dari:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Jumlah baris & kolom:", df.shape)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    print("Missing value per kolom:")
    print(X.isnull().sum())

    # Split SEBELUM fit preprocessing — mencegah data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print("Jumlah data train:", X_train.shape[0])
    print("Jumlah data test :", X_test.shape[0])
    print("Proporsi churn train:\n", y_train.value_counts(normalize=True).round(3))
    print("Proporsi churn test :\n", y_test.value_counts(normalize=True).round(3))

    preprocessor = build_preprocessor()

    # fit_transform HANYA di train, transform saja di test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Bentuk data setelah preprocessing:", X_train_processed.shape)

    feature_names = preprocessor.get_feature_names_out()
    print("Total fitur setelah preprocessing:", len(feature_names))

    # =========================================================
    # Simpan semua output ke OUTPUT_DIR (= src/)
    # =========================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.pkl"))
    np.save(os.path.join(OUTPUT_DIR, "X_train_processed.npy"), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, "X_test_processed.npy"), X_test_processed)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    print("\nSelesai. File tersimpan di", OUTPUT_DIR, ":")
    print("- preprocessor.pkl")
    print("- X_train_processed.npy, X_test_processed.npy")
    print("- y_train.csv, y_test.csv")
    print("\nLangkah selanjutnya: jalankan  python src/train.py")


if __name__ == "__main__":
    main()