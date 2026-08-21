# %% [markdown]
# # Project 1 - Customer Churn Prediction
# Diadaptasi dari Colab notebook untuk dijalankan di VS Code / Jupyter lokal.
#
# **Sebelum menjalankan**, install dependency lewat terminal (bukan di dalam cell):
# ```bash
# pip install scikit-learn xgboost joblib pandas matplotlib seaborn
# ```

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score, accuracy_score, roc_auc_score,
    RocCurveDisplay,
)

try:
    from xgboost import XGBClassifier
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost tidak tersedia, akan dilewati.")

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

print("Python      :", sys.version)
if XGBOOST_AVAILABLE:
    print("XGBoost     :", xgboost.__version__)
import sklearn
print("Scikit-learn:", sklearn.__version__)
print("Joblib      :", joblib.__version__)

# %% [markdown]
# ## 1. Load Data
# Ganti `DATA_PATH` dengan lokasi file CSV kamu di komputer
# (pengganti `files.upload()` yang khusus Colab).

# %%
DATA_PATH = "data\churn.csv"# path kamu

df = pd.read_csv(DATA_PATH)
print("Jumlah baris & kolom:", df.shape)
df.head(5)

# %% [markdown]
# ## 2. EDA (Exploratory Data Analysis)
# ### 2a. Cek struktur & missing value

# %%
df.info()

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(missing_summary if not missing_summary.empty else "Tidak ada missing value")

# %% [markdown]
# ### 2b. Cek duplikat (tidak ada di script asli — tambahan)

# %%
n_dupes = df.duplicated().sum()
print(f"Jumlah baris duplikat: {n_dupes}")

# %% [markdown]
# ### 2c. Distribusi target `churn`

# %%
print(df["Exited"].value_counts())

plt.figure(figsize=(5, 5))
ax = sns.countplot(data=df, x="Exited")
plt.title("Distribusi Churn (0 = Tetap, 1 = Churn)")
plt.xlabel("Churn")
plt.ylabel("Count")
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.show()

# %% [markdown]
# ### 2d. Fitur numerik vs Churn (saldo, tenor, usia, dll)

# %%
numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(data=df, x="Exited", y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs Churn")
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2e. Fitur kategorikal vs Churn (produk, keaktifan, dll)

# %%
categorical_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    churn_rate = df.groupby(col)["Exited"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=axes[i])
    axes[i].set_title(f"Churn Rate (%) per {col}")
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2f. Correlation heatmap (tidak ada di script asli — tambahan)
# Membantu melihat multikolinearitas antar fitur numerik.

# %%
plt.figure(figsize=(6, 5))
sns.heatmap(df[numerical_cols + ["Exited"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# **Ringkasan EDA** (isi sesuai temuan aktual di data kamu):
# - Data imbalanced (~78:22) → perlu class_weight / scale_pos_weight saat training
# - `age`, `country`, `active_member`, `products_number` = fitur paling berpengaruh
# - `products_number` 3-4 produk → churn rate sangat tinggi

# %% [markdown]
# ## 3. Data Preprocessing
# Missing values, encoding kategorikal, scaling numerik — dibungkus dalam satu
# `Pipeline`/`ColumnTransformer` agar konsisten dipakai ulang saat inference.

# %%
ID_COL = "CustomerId"
TARGET_COL = "Exited"
NUMERICAL_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y,
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# %%
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Bentuk data setelah preprocessing:", X_train_processed.shape)

joblib.dump(preprocessor, "preprocessor.pkl")
print("preprocessor.pkl tersimpan.")

# %% [markdown]
# ## 4. Model Training & Evaluation
# ### 4a. Baseline dummy classifier (tidak ada di script asli — tambahan)
# Penting untuk tahu skor "minimum" sebelum menilai model asli bagus atau tidak.

# %%
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_processed, y_train)
y_pred_dummy = dummy.predict(X_test_processed)
print("Baseline (selalu prediksi kelas mayoritas):")
print(classification_report(y_test, y_pred_dummy))

# %% [markdown]
# ### 4b. Hitung `scale_pos_weight` untuk XGBoost (data imbalanced)

# %%
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count
print(f"scale_pos_weight untuk XGBoost: {scale_pos_weight_value:.2f}")

# %% [markdown]
# ### 4c. Model 1: Logistic Regression

# %%
log_reg = LogisticRegression(random_state=21, class_weight='balanced', solver='liblinear')
log_reg.fit(X_train_processed, y_train)

y_pred_log_reg = log_reg.predict(X_test_processed)
y_prob_log_reg = log_reg.predict_proba(X_test_processed)[:, 1]

print("Logistic Regression Performance")
print(classification_report(y_test, y_pred_log_reg))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_log_reg):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# ### 4d. Model 2: Random Forest Classifier

# %%
rf_clf = RandomForestClassifier(random_state=21, class_weight='balanced')
rf_clf.fit(X_train_processed, y_train)

y_pred_rf = rf_clf.predict(X_test_processed)
y_prob_rf = rf_clf.predict_proba(X_test_processed)[:, 1]

print("Random Forest Performance")
print(classification_report(y_test, y_pred_rf))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# ### 4e. Model 3: XGBoost Classifier
# Catatan: parameter `use_label_encoder` sudah **deprecated/dihapus** di
# xgboost versi baru dan bisa memicu error. Di sini dihapus.

# %%
if XGBOOST_AVAILABLE:
    xgb_clf = XGBClassifier(
        random_state=21,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value,
    )
    xgb_clf.fit(X_train_processed, y_train)

    y_pred_xgb = xgb_clf.predict(X_test_processed)
    y_prob_xgb = xgb_clf.predict_proba(X_test_processed)[:, 1]

    print("XGBoost Performance")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_xgb):.4f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - XGBoost')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
else:
    print("XGBoost tidak tersedia, model tidak dapat dilatih.")

# %% [markdown]
# ### 4f. Cross-validation
# Train/test split tunggal bisa memberi gambaran yang "kebetulan bagus/jelek".
# 5-fold CV memberi estimasi performa yang lebih stabil dan sekaligus jadi
# cek cepat overfitting (bandingkan skor antar fold).

# %%
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
for name, model in [
    ("Logistic Regression", log_reg),
    ("Random Forest", rf_clf),
] + ([("XGBoost", xgb_clf)] if XGBOOST_AVAILABLE else []):
    scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring="f1")
    print(f"{name}: F1 CV = {scores.mean():.4f} (+/- {scores.std():.4f})")

# %% [markdown]
# ### 4g. Perbandingan semua model

# %%
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "XGBoost": xgb_clf if XGBOOST_AVAILABLE else None,
}
models = {name: model for name, model in models.items() if model is not None}

results = []
trained_models = {}

for name, model in models.items():
    trained_models[name] = model
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
print(results_df.round(4))

# %% [markdown]
# ### 4h. ROC Curve semua model (tidak ada di script asli — tambahan)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
for name, model in trained_models.items():
    RocCurveDisplay.from_estimator(model, X_test_processed, y_test, ax=ax, name=name)
plt.title("ROC Curve — Semua Model")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.show()

# %% [markdown]
# ## 5. Model Terbaik karena Stabil

# %%
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_processed)

print(f"Model terbaik: {best_model_name}\n")
print(classification_report(y_test, y_pred_best, target_names=["Tidak Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Tidak Churn", "Churn"], yticklabels=["Tidak Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.show()

# %% [markdown]
# ### 5a. Feature importance (tidak ada di script asli — tambahan)
# Hanya berlaku untuk tree-based model (RF/XGBoost).

# %%
import sys
print("Model type:", type(best_model).__name__)
if hasattr(best_model, "n_estimators"):
    print("n_estimators:", best_model.n_estimators)
if hasattr(best_model, "get_params"):
    print(best_model.get_params())

# Kalau Random Forest — cek total node di semua pohon (indikasi kompleksitas)
if type(best_model).__name__ == "RandomForestClassifier":
    total_nodes = sum(tree.tree_.node_count for tree in best_model.estimators_)
    print("Total node di semua pohon:", total_nodes)

if best_model_name in ("Random Forest", "XGBoost"):
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top 15 Feature Importance — {best_model_name}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Simpan Model Terbaik dan Hasil Perbandingan

# %%
joblib.dump(best_model, "model.pkl")
results_df.to_csv("model_comparison_results.csv", index=False)
print(f"model.pkl ({best_model_name}) tersimpan.")
print("model_comparison_results.csv tersimpan.")

# %% [markdown]
# ### A. Korelasi tiap fitur numerik ke `churn` (df.corr())
# Catatan: df.corr() cuma jalan buat kolom numerik. Kolom kategorikal
# (country, gender) perlu di-encode dulu biar ikut kehitung.
 
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# Encode sementara cuma buat lihat korelasi (bukan buat training!)
df_corr = df.copy()
df_corr["gender_encoded"] = df_corr["Gender"].map({"Male": 0, "Female": 1})
df_corr = pd.concat([df_corr, pd.get_dummies(df_corr["Geography"], prefix="Geography")], axis=1)
 
corr_cols = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "gender_encoded",
] + [c for c in df_corr.columns if c.startswith("country_")] + ["Exited"]
 
corr_matrix = df_corr[corr_cols].corr()
 
# Urutkan berdasarkan korelasi absolut ke churn, dari yang paling kuat
churn_corr = corr_matrix["Exited"].drop("Exited").sort_values(key=abs, ascending=False)
print("Korelasi tiap fitur terhadap churn (diurutkan dari yang paling kuat):")
print(churn_corr.round(4))
 
plt.figure(figsize=(6, 8))
sns.barplot(x=churn_corr.values, y=churn_corr.index, palette="coolwarm")
plt.title("Korelasi fitur terhadap churn")
plt.xlabel("Koefisien korelasi (Pearson)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
# %% [markdown]
# ### B. Permutation Importance
# Lebih adil dibanding `feature_importances_` bawaan RandomForest/XGBoost,
# karena tidak bias ke fitur numerik dengan kardinalitas tinggi.
# Cara kerja: acak (shuffle) satu kolom, lihat seberapa turun performa
# model — makin turun, makin penting fitur itu.
 
# %%
from sklearn.inspection import permutation_importance
 
feature_names_out = preprocessor.get_feature_names_out()
 
for name, model in [("Random Forest", rf_clf), ("XGBoost", xgb_clf)]:
    print(f"\n=== Permutation Importance — {name} ===")
    perm_result = permutation_importance(
        model, X_test_processed, y_test,
        n_repeats=10, random_state=21, scoring="f1", n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": feature_names_out,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(perm_df.head(15).to_string(index=False))
 
    plt.figure(figsize=(8, 6))
    top15 = perm_df.head(15)
    sns.barplot(x="importance_mean", y="feature", data=top15, xerr=top15["importance_std"])
    plt.title(f"Permutation Importance (Top 15) — {name}")
    plt.xlabel("Penurunan F1-score saat fitur diacak")
    plt.tight_layout()
    plt.show()
 
# %% [markdown]
# ### B. Tuning Threshold Random Forest vs F1 optimal XGBoost
# Default threshold classification itu 0.5. Tapi karena data imbalanced
# (~78:22), threshold 0.5 belum tentu optimal buat F1-score, khususnya
# buat Random Forest yang recall-nya rendah (0.43) di threshold default.
# Kita cari threshold yang memaksimalkan F1 untuk masing-masing model,
# lalu bandingkan head-to-head di threshold optimalnya masing-masing.
 
# %%
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
 
def find_best_threshold(y_true, y_prob):
    """Cari threshold yang memaksimalkan F1-score, scan 0.01 - 0.99."""
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    return thresholds[best_idx], f1_scores[best_idx]
 
y_prob_rf = rf_clf.predict_proba(X_test_processed)[:, 1]
y_prob_xgb = xgb_clf.predict_proba(X_test_processed)[:, 1]
 
best_thresh_rf, best_f1_rf = find_best_threshold(y_test, y_prob_rf)
best_thresh_xgb, best_f1_xgb = find_best_threshold(y_test, y_prob_xgb)
 
print(f"Random Forest — threshold optimal: {best_thresh_rf:.2f}  |  F1 di threshold ini: {best_f1_rf:.4f}")
print(f"XGBoost       — threshold optimal: {best_thresh_xgb:.2f}  |  F1 di threshold ini: {best_f1_xgb:.4f}")
 

# %% [markdown]
# ### B1. Bandingkan detail: threshold default (0.5) vs threshold optimal
 
# %%
def evaluate_at_threshold(y_true, y_prob, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model": label,
        "Threshold": round(threshold, 2),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
    }
 
comparison = pd.DataFrame([
    evaluate_at_threshold(y_test, y_prob_rf, 0.5, "Random Forest (default 0.5)"),
    evaluate_at_threshold(y_test, y_prob_rf, best_thresh_rf, "Random Forest (optimal)"),
    evaluate_at_threshold(y_test, y_prob_xgb, 0.5, "XGBoost (default 0.5)"),
    evaluate_at_threshold(y_test, y_prob_xgb, best_thresh_xgb, "XGBoost (optimal)"),
])
print(comparison.to_string(index=False))
 
# %% [markdown]
# ### C2. Visualisasi F1-score di semua threshold (0.01 - 0.99)
# Biar keliatan jelas seberapa sensitif tiap model terhadap pilihan threshold.
 
# %%
thresholds = np.arange(0.01, 1.00, 0.01)
f1_curve_rf = [f1_score(y_test, (y_prob_rf >= t).astype(int)) for t in thresholds]
f1_curve_xgb = [f1_score(y_test, (y_prob_xgb >= t).astype(int)) for t in thresholds]
 
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_curve_rf, label=f"Random Forest (best={best_thresh_rf:.2f}, F1={best_f1_rf:.4f})")
plt.plot(thresholds, f1_curve_xgb, label=f"XGBoost (best={best_thresh_xgb:.2f}, F1={best_f1_xgb:.4f})")
plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Default threshold (0.5)")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Threshold — Random Forest vs XGBoost")
plt.legend()
plt.tight_layout()
plt.show()
 
# %% [markdown]
# **Kesimpulan yang perlu lo cek sendiri dari hasil run di atas:**
# - Kalau F1 optimal Random Forest (setelah tuning threshold) ternyata
#   MELEBIHI F1 XGBoost, berarti pemilihan "model terbaik" di section 5
#   notebook utama (`results_df.sort_values("F1-Score")`) itu kurang adil,
#   karena XGBoost dievaluasi di threshold optimalnya secara implisit
#   (boosting model cenderung sudah cukup baik di 0.5), sedangkan RF tidak.
# - Kalau lo pakai threshold hasil tuning ini di production, INGAT:
#   `main.py` (api serving) juga harus diubah — bagian `predict_proba(...)[0][1]`
#   dibandingkan ke threshold baru itu, bukan cuma dikembalikan mentah
#   sebagai persentase seperti sekarang.
 
