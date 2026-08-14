# %% [markdown]
# # Model Training & Evaluation — Bank Customer Churn
#
# Script ini lanjutan dari `preprocessing_bank_churn.py`. Pastikan sudah
# dijalankan duluan supaya file-file berikut sudah ada di folder yang sama:
# - `X_train_processed.npy`, `X_test_processed.npy`
# - `y_train.csv`, `y_test.csv`
# - `preprocessor.pkl`
#
# Tujuan: melatih beberapa model, membandingkan performanya (bukan cuma
# akurasi karena data imbalanced), lalu simpan model terbaik untuk dipakai
# di tahap Model Serving (API).

# %%
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
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ xgboost belum terinstall. Jalankan: pip install xgboost")
    print("   Sementara ini XGBoost akan dilewati.")

INPUT_DIR = "."  # folder tempat file hasil preprocessing berada

# %% [markdown]
# ## 1. Load Data Hasil Preprocessing

# %%
X_train = np.load(f"{INPUT_DIR} /C:\Users\ASUS\Downloads\credit-scoring-project\X_train_processed.npy")
X_test = np.load(f"{INPUT_DIR}/C:\Users\ASUS\Downloads\credit-scoring-project\X_test_processed.npy")
y_train = pd.read_csv(f"{INPUT_DIR}/C:\Users\ASUS\Downloads\credit-scoring-project\y_train.csv").values.ravel()
y_test = pd.read_csv(f"{INPUT_DIR}/C:\Users\ASUS\Downloads\credit-scoring-project\y_test.csv").values.ravel()

print("Bentuk X_train:", X_train.shape)
print("Bentuk X_test:", X_test.shape)

# %% [markdown]
# ## 2. Kenapa Tidak Boleh Cuma Pakai Akurasi?
#
# Data kita imbalanced (~78% tidak churn, ~22% churn). Model yang asal
# nebak "semua nasabah tidak churn" bisa dapat akurasi ~78% tanpa belajar
# apa-apa — makanya kita fokus ke metrik berikut:
#
# - **Recall**: dari semua nasabah yang BENERAN churn, berapa persen yang
#   berhasil model tangkap? Ini penting karena kalau nasabah churn tidak
#   terdeteksi, tim CRM kehilangan kesempatan untuk retensi.
# - **Precision**: dari semua yang model bilang "bakal churn", berapa
#   persen yang benar? Penting supaya tim CRM tidak buang waktu follow-up
#   nasabah yang sebenarnya tidak akan churn.
# - **F1-Score**: rata-rata harmonik Recall & Precision, jadi metrik
#   penyeimbang antara keduanya.

# %% [markdown]
# ## 3. Definisikan Model-Model yang Akan Dibandingkan
#
# `class_weight="balanced"` dipakai di Logistic Regression & Random Forest
# supaya model tidak bias ke kelas mayoritas (tidak churn), tanpa perlu
# oversampling data secara manual (misal SMOTE).

# %%
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=200, random_state=42
    ),
}

if XGBOOST_AVAILABLE:
    # scale_pos_weight = rasio kelas mayoritas/minoritas, fungsinya mirip class_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models["XGBoost"] = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

print("Model yang akan dilatih:", list(models.keys()))

# %% [markdown]
# ## 4. Training + Cross-Validation
#
# Cross-validation (5-fold) dipakai supaya evaluasi lebih robust, bukan
# cuma mengandalkan satu kali split train-test yang bisa kebetulan
# "beruntung" atau "apes". `StratifiedKFold` menjaga proporsi churn tetap
# seimbang di setiap fold.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_results[name] = f1_scores
    print(f"{name}: F1-Score CV = {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

# %% [markdown]
# ## 5. Fit Model di Seluruh Data Train, lalu Evaluasi di Data Test
#
# Cross-validation di atas cuma untuk sanity-check awal. Evaluasi final
# tetap harus di data test yang belum pernah dilihat model sama sekali.

# %%
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
print(results_df.round(4))

# %% [markdown]
# ## 6. Visualisasi Perbandingan Model

# %%
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

# %% [markdown]
# ## 7. Detail Model Terbaik — Classification Report & Confusion Matrix
#
# Dipilih otomatis berdasarkan F1-Score tertinggi (bisa diganti manual
# kalau tim mau pertimbangkan metrik lain, misal Recall lebih diprioritaskan).

# %%
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"=== Model Terbaik: {best_model_name} ===\n")
print(classification_report(y_test, y_pred_best, target_names=["Tidak Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Tidak Churn", "Churn"],
            yticklabels=["Tidak Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Aktual")
plt.xlabel("Prediksi")
plt.show()

# %% [markdown]
# ## 8. Feature Importance (Khusus Random Forest / XGBoost)
#
# Berguna untuk validasi ulang: apakah fitur yang dianggap penting oleh
# model cocok dengan insight yang kita temukan waktu EDA
# (`age`, `country`, `active_member`, `products_number`)?

# %%
if best_model_name in ["Random Forest", "XGBoost"]:
    # Ambil nama fitur dari preprocessor yang disimpan waktu preprocessing
    preprocessor = joblib.load(f"{INPUT_DIR}/C:\Users\ASUS\Downloads\credit-scoring-project\preprocessor.pkl")
    feature_names = preprocessor.get_feature_names_out()

    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top 10 Feature Importance — {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Feature importance hanya ditampilkan untuk Random Forest/XGBoost.")

# %% [markdown]
# ## 9. Simpan Model Terbaik
#
# File `model.pkl` ini, bersama `preprocessor.pkl` dari tahap sebelumnya,
# adalah dua file yang dibutuhkan di tahap Model Serving (FastAPI).

# %%
joblib.dump(best_model, f"{INPUT_DIR}/model.pkl")

results_df.to_csv(f"{INPUT_DIR}/model_comparison_results.csv", index=False)

print(f"Model terbaik ({best_model_name}) disimpan sebagai 'model.pkl'")
print("Hasil perbandingan semua model disimpan sebagai 'model_comparison_results.csv'")

# %% [markdown]
# ## Catatan Lanjutan
#
# - Kalau `Recall` model terbaik masih dirasa kurang tinggi (banyak nasabah
#   churn yang tidak terdeteksi), coba turunkan threshold prediksi default
#   (0.5) jadi lebih rendah, misal 0.35 — ini trade-off dengan Precision,
#   perlu didiskusikan bareng tim bisnis/CRM mana yang lebih diprioritaskan.
# - `model.pkl` dan `preprocessor.pkl` akan dipakai bersama-sama di endpoint
#   FastAPI: data masuk → `preprocessor.transform()` → `model.predict_proba()`.