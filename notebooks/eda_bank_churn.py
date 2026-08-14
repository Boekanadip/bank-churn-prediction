# %% [markdown]
# # EDA — Bank Customer Churn Dataset
#
# Script ini dibagi jadi beberapa "cell" (dipisah `# %%`) supaya bisa dijalankan
# per bagian di VS Code / Jupyter / Spyder, atau langsung full run sebagai script biasa.
#
# Ganti `DATA_PATH` di bawah sesuai lokasi file CSV hasil download dari Kaggle.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_PATH = r"C:\Users\ASUS\Downloads\fakedata\Bank Customer Churn Prediction.csv"  # <-- sesuaikan nama/path file lo

df = pd.read_csv(DATA_PATH)
print("Jumlah baris & kolom:", df.shape)
df.head()

# %% [markdown]
# ## 1. Overview Dataset
# Cek tipe data, jumlah non-null, dan statistik ringkas dulu sebelum masuk lebih dalam.

# %%
df.info()

# %%
df.describe(include="all").T

# %% [markdown]
# ## 2. Cek Missing Value
# Kalau ada kolom dengan missing value, kita perlu tau seberapa parah dan
# apakah pola-nya random atau ada alasan tertentu (misal berkaitan sama nasabah lama).

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values(
    "missing_pct", ascending=False
)
print("Kolom dengan missing value:")
print(missing_summary if not missing_summary.empty else "Tidak ada missing value 🎉")

# %% [markdown]
# ## 3. Distribusi Target (Churn)
# Ini penting banget — kalau imbalanced (misal churn cuma 20% dari total data),
# kita perlu strategi khusus nanti pas training (class_weight, SMOTE, dll).
# Jangan langsung pakai akurasi sebagai metrik utama kalau imbalanced.

# %%
churn_counts = df["churn"].value_counts()
churn_pct = df["churn"].value_counts(normalize=True) * 100

print("Jumlah per kelas:\n", churn_counts)
print("\nPersentase per kelas:\n", churn_pct.round(2))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="churn")
plt.title("Distribusi Churn (0 = Tetap, 1 = Churn)")
plt.xlabel("Churn")
plt.ylabel("Jumlah Nasabah")
plt.show()

# %% [markdown]
# ## 4. Distribusi Fitur Numerik
# Lihat sebaran tiap fitur numerik — cek juga kalau ada outlier ekstrem
# (misal balance yang aneh tinggi atau age yang nggak masuk akal).

# %%
numerical_cols = ["credit_score", "age", "tenure", "balance", "estimated_salary"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribusi {col}")
fig.delaxes(axes[-1])  # buang subplot kosong terakhir
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Fitur Numerik vs Churn
# Ini bagian intinya EDA buat cari pola: apakah nasabah yang churn punya
# karakteristik numerik yang beda dari yang nggak churn?
# Pakai boxplot supaya kelihatan perbedaan median & sebarannya.

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(data=df, x="churn", y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs Churn")
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Fitur Kategorikal vs Churn
# Cek apakah negara, gender, kepemilikan kartu kredit, atau status aktif
# member punya kecenderungan churn yang beda-beda.

# %%
categorical_cols = ["country", "gender", "credit_card", "active_member", "products_number"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=axes[i])
    axes[i].set_title(f"Churn Rate (%) per {col}")
    axes[i].set_ylabel("Churn Rate (%)")
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Correlation Matrix (Fitur Numerik)
# Lihat hubungan antar fitur numerik, termasuk korelasinya ke churn.
# Perhatikan juga kalau ada fitur yang saling berkorelasi tinggi satu sama lain
# (multicollinearity) — bisa jadi pertimbangan nanti pas feature selection.

# %%
corr_cols = numerical_cols + ["churn"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# ## 8. Ringkasan Insight (isi manual setelah lihat hasil di atas)
#
# Contoh pertanyaan yang perlu dijawab setelah run semua cell di atas:
#
# - [ ] Apakah data imbalanced? Seberapa parah?
# - [ ] Negara mana yang churn rate-nya paling tinggi?
# - [ ] Apakah nasabah yang tidak aktif (`active_member = 0`) lebih banyak churn?
# - [ ] Apakah usia berkorelasi dengan churn (makin tua makin/less churn)?
# - [ ] Apakah jumlah produk (`products_number`) berpengaruh ke churn?
# - [ ] Ada fitur yang perlu di-drop karena nyaris tidak ada hubungan ke churn?
#
# Tulis 3-5 insight kunci di sini setelah selesai analisis — ini yang nanti
# dipakai buat pertimbangan feature engineering di tahap preprocessing.

# %%
