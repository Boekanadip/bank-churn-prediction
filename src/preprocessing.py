# preprocessing
# %% [markdown]
# # Data Preprocessing — Bank Customer Churn Dataset
#
# Tujuan: siapin data mentah supaya bisa langsung dipakai training model.
# Ada 3 hal utama yang dikerjain di sini:
# 1. Menangani missing values
# 2. Encoding variabel kategorikal (country, gender)
# 3. Scaling fitur numerik
#
# Semua langkah ini dibungkus jadi satu **Pipeline** (bukan dikerjain manual
# satu-satu), supaya nanti gampang dipanggil ulang persis sama di API
# (main.py) tanpa perlu nulis ulang logic preprocessing dari nol.

# %%
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = r"C:\Users\ASUS\Downloads\fakedata\Bank Customer Churn Prediction.csv"  # <-- sesuaikan path file lo
OUTPUT_DIR = "."  # folder buat nyimpen hasil preprocessing & pipeline

df = pd.read_csv(DATA_PATH)
print("Jumlah baris & kolom:", df.shape)
df.head()

# %% [markdown]
# ## 1. Pisahkan Kolom Berdasarkan Perannya
#
# - `customer_id` → disimpan terpisah, **tidak** dipakai sebagai fitur model
#   (sesuai catatan dataset: "unused variable"), tapi tetap dibutuhkan nanti
#   buat dikembalikan lagi di response API.
# - `churn` → target/label yang mau diprediksi.
# - Sisanya → fitur input model, dibagi lagi jadi numerik & kategorikal
#   karena keduanya butuh perlakuan preprocessing yang beda.

# %%
id_col = "customer_id"
target_col = "churn"

numerical_features = [
    "credit_score", "age", "tenure", "balance",
    "products_number", "estimated_salary",
]
categorical_features = ["country", "gender", "credit_card", "active_member"]

X = df[numerical_features + categorical_features]
y = df[target_col]

print("Fitur numerik:", numerical_features)
print("Fitur kategorikal:", categorical_features)
print("Target:", target_col)

# %% [markdown]
# ## 2. Cek Missing Values Lagi (Sebelum Diproses)
#
# Dataset ini kemungkinan besar sudah bersih (hasil cek EDA sebelumnya),
# tapi kita tetap bikin langkah penanganan missing value supaya pipeline ini
# **tetap aman dipakai** kalau nanti datanya diganti data asli bank yang
# mungkin ada bolongnya.

# %%
print("Missing value per kolom:")
print(X.isnull().sum())

# %% [markdown]
# ## 3. Bangun Pipeline Preprocessing
#
# Kenapa pakai Pipeline + ColumnTransformer, bukan diproses manual pakai
# pandas satu-satu?
# - Supaya langkah imputasi, encoding, dan scaling **konsisten** dipakai ulang
#   persis sama saat training maupun saat serving lewat API (menghindari
#   bug "beda perlakuan antara training dan production").
# - Bisa disimpan jadi satu file (`preprocessor.pkl`) dan tinggal di-load,
#   tidak perlu nulis ulang logic-nya di kode API.

# %%
# --- Sub-pipeline untuk fitur numerik ---
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # isi missing value pakai median (tahan outlier)
    ("scaler", StandardScaler()),                     # scaling supaya semua fitur di skala sebanding
])

# --- Sub-pipeline untuk fitur kategorikal ---
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # isi missing value pakai nilai paling sering muncul
    ("onehot", OneHotEncoder(handle_unknown="ignore")),      # ubah teks jadi angka biner (0/1)
])

# --- Gabungkan keduanya dalam satu ColumnTransformer ---
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

# %% [markdown]
# ## 4. Split Data (Train & Test) SEBELUM Fit Preprocessing
#
# **Penting:** pemisahan train/test harus dilakukan SEBELUM `fit()`
# preprocessing. Kalau kebalik (fit dulu baru split), informasi dari data
# test bisa "bocor" ke proses training (disebut *data leakage*) dan bikin
# evaluasi model nanti kelihatan bagus padahal di dunia nyata belum tentu.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # penting karena data imbalanced — jaga proporsi churn tetap sama di train & test
)

print("Jumlah data train:", X_train.shape[0])
print("Jumlah data test:", X_test.shape[0])
print("\nProporsi churn di train:\n", y_train.value_counts(normalize=True).round(3))
print("\nProporsi churn di test:\n", y_test.value_counts(normalize=True).round(3))

# %% [markdown]
# ## 5. Fit Preprocessing HANYA di Data Train
#
# `fit_transform()` dipakai di data train (belajar sekaligus transform).
# `transform()` saja (tanpa fit) dipakai di data test — supaya statistik
# yang dipelajari (misal rata-rata & standar deviasi untuk scaling) murni
# berasal dari data train, bukan "mengintip" data test.

# %%
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Bentuk data train setelah preprocessing:", X_train_processed.shape)
print("Bentuk data test setelah preprocessing:", X_test_processed.shape)

# %% [markdown]
# ## 6. Cek Nama Kolom Hasil Encoding (Opsional, buat Verifikasi)
#
# Berguna untuk mastiin one-hot encoding menghasilkan kolom yang sesuai
# ekspektasi, sebelum lanjut ke tahap training.

# %%
feature_names = preprocessor.get_feature_names_out()
print("Total fitur setelah preprocessing:", len(feature_names))
print(feature_names)

# %% [markdown]
# ## 7. Simpan Pipeline Preprocessing & Data yang Sudah Diproses
#
# `preprocessor.pkl` ini nanti **dipakai lagi persis sama** di kode API
# (FastAPI), supaya data yang dikirim tim Laravel diproses dengan cara yang
# identik dengan waktu training. Ini kunci utama biar hasil prediksi konsisten.

# %%
joblib.dump(preprocessor, f"{OUTPUT_DIR}/preprocessor.pkl")

np.save(f"{OUTPUT_DIR}/X_train_processed.npy", X_train_processed)
np.save(f"{OUTPUT_DIR}/X_test_processed.npy", X_test_processed)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

print("Selesai. File tersimpan:")
print("- preprocessor.pkl       (pipeline, dipakai ulang di API)")
print("- X_train_processed.npy  (siap dipakai training)")
print("- X_test_processed.npy   (siap dipakai evaluasi)")
print("- y_train.csv, y_test.csv")

# %% [markdown]
# ## Catatan Lanjutan
#
# - File `preprocessor.pkl` adalah salah satu dari 2 file yang perlu dibawa
#   ke tahap Model Serving nanti, satu lagi adalah `model.pkl` hasil training.
# - Kalau nanti dataset diganti data asli dari bank dengan kolom yang beda,
#   cukup update daftar `numerical_features` dan `categorical_features` di
#   atas — struktur pipeline-nya tidak perlu diubah.
# %%
