"""
src/train.py

Training & evaluasi model — dijalankan SETELAH src/preprocessing.py.

Cara jalanin (dari root folder project):
    python src/train.py

Prasyarat: file-file berikut sudah ada di folder src/ (dihasilkan oleh
preprocessing.py):
    - X_train_processed.npy, X_test_processed.npy
    - y_train.csv, y_test.csv
    - preprocessor.pkl

Output:
    - src/model.pkl                  (model dengan F1-Score terbaik)
    - src/metadata.json              (versi library, metrik, rentang fitur)
    - model_comparison_results.csv   (di root project, buat referensi)
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score, accuracy_score, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost belum terinstall. Jalankan: pip install xgboost")

import sklearn

# Import definisi kolom dari preprocessing.py, supaya tidak ada duplikasi
# definisi fitur di dua tempat berbeda.
from preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# =========================================================
# Konfigurasi path — semua relatif terhadap lokasi file ini
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../bank-churn-prediction/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                 # .../bank-churn-prediction
INPUT_DIR = BASE_DIR   # baca hasil preprocessing dari src/
OUTPUT_DIR = BASE_DIR  # simpan model.pkl & metadata.json ke src/ juga

print("Python      :", sys.version.split()[0])
print("Scikit-learn:", sklearn.__version__)
if XGBOOST_AVAILABLE:
    print("XGBoost     :", xgboost.__version__)
print("Joblib      :", joblib.__version__)
print("Numpy       :", np.__version__)

# =========================================================
# 1. Load data hasil preprocessing
# =========================================================
X_train = np.load(os.path.join(INPUT_DIR, "X_train_processed.npy"), allow_pickle=True)
X_test = np.load(os.path.join(INPUT_DIR, "X_test_processed.npy"), allow_pickle=True)
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv")).values.ravel()

print("\nBentuk X_train:", X_train.shape)
print("Bentuk X_test :", X_test.shape)

# =========================================================
# 2. Definisikan model yang dibandingkan
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, solver="liblinear",
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=200, random_state=42,
    ),
}

if XGBOOST_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models["XGBoost"] = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )

print("\nModel yang akan dilatih:", list(models.keys()))

# =========================================================
# 3. Cross-validation (sanity check awal)
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nCross-validation (F1-Score, 5-fold):")
for name, model in models.items():
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"{name}: F1 CV = {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

# =========================================================
# 4. Fit di seluruh data train, evaluasi final di data test
# =========================================================
results_summary = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    trained_models[name] = model
    results_summary.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })

results_df = pd.DataFrame(results_summary).sort_values("F1-Score", ascending=False)
print("\nPerbandingan semua model:")
print(results_df.round(4))

# =========================================================
# 5. Visualisasi perbandingan
# =========================================================
metrics_to_plot = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
results_melted = results_df.melt(id_vars="Model", value_vars=metrics_to_plot,
                                   var_name="Metric", value_name="Score")

plt.figure(figsize=(9, 5))
sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model")
plt.title("Perbandingan Metrik Antar Model")
plt.ylim(0, 1)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================================================
# 6. Detail model terbaik
# =========================================================
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"\n=== Model Terbaik: {best_model_name} ===\n")
print(classification_report(y_test, y_pred_best, target_names=["Tidak Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Tidak Churn", "Churn"], yticklabels=["Tidak Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Aktual")
plt.xlabel("Prediksi")
plt.show()

# =========================================================
# 7. Feature importance (khusus Random Forest / XGBoost)
# =========================================================
if best_model_name in ["Random Forest", "XGBoost"]:
    preprocessor = joblib.load(os.path.join(INPUT_DIR,"preprocessor.pkl"))
    feature_names = preprocessor.get_feature_names_out()

    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top 15 Feature Importance — {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Feature importance hanya ditampilkan untuk Random Forest/XGBoost.")

# =========================================================
# 8. Simpan model terbaik + metadata
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "model.pkl"))
results_df.to_csv(os.path.join(PROJECT_ROOT,   "model_comparison_results.csv"), index=False)

# Hitung rentang nilai valid tiap fitur numerik dari data mentah,
# untuk didokumentasikan di metadata.json (referensi validasi input di API)
raw_data_path = os.path.join(PROJECT_ROOT, "data", "churn.csv")
df_raw = pd.read_csv(raw_data_path)
feature_ranges = {
    col: [float(df_raw[col].min()), float(df_raw[col].max())]
    for col in NUMERICAL_FEATURES
}

best_metrics = results_df.iloc[0].to_dict()
metadata = {
    "model_name": best_model_name,
    "model_version": f"v1.0-{best_model_name.lower().replace(' ', '_')}",
    "trained_at": datetime.date.today().isoformat(),
    "library_versions": {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__ if XGBOOST_AVAILABLE else None,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
    },
    "features": {
        "numerical": NUMERICAL_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "feature_valid_range": feature_ranges,
    "metrics": {
        "accuracy": round(best_metrics["Accuracy"], 4),
        "precision": round(best_metrics["Precision"], 4),
        "recall": round(best_metrics["Recall"], 4),
        "f1_score": round(best_metrics["F1-Score"], 4),
        "roc_auc": round(best_metrics["ROC-AUC"], 4),
    },
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel terbaik ({best_model_name}) disimpan sebagai src/model.pkl")
print("Metadata disimpan sebagai src/metadata.json")
print("Hasil perbandingan semua model disimpan sebagai model_comparison_results.csv")
print("\nSelesai. Sekarang jalankan API dengan: uvicorn api.main:app --reload")