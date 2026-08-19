# Bank Customer Churn Prediction

Project Machine Learning untuk memprediksi kemungkinan **customer churn** pada nasabah bank berdasarkan karakteristik demografi, kondisi finansial, dan aktivitas nasabah.

Project ini mencakup proses **Data Science end-to-end**, mulai dari data understanding, data cleaning, exploratory data analysis (EDA), preprocessing, pelatihan dan evaluasi model Machine Learning, hingga implementasi model sebagai REST API menggunakan **FastAPI**.

---

## Tujuan Project

Customer churn merupakan kondisi ketika nasabah berhenti menggunakan layanan atau produk suatu bank.

Tujuan dari project ini adalah membangun model klasifikasi yang dapat:

* Memprediksi apakah seorang nasabah berpotensi melakukan churn.
* Menghasilkan probabilitas churn.
* Mengelompokkan tingkat risiko churn berdasarkan probabilitas prediksi.
* Menyediakan hasil prediksi melalui API agar dapat diintegrasikan dengan aplikasi lain.

Model Machine Learning yang telah dilatih kemudian digunakan oleh FastAPI sebagai layanan prediksi yang dapat dikonsumsi oleh software atau backend application.

---

## Alur Project

```text
Data
  │
  ▼
Data Understanding
  │
  ▼
Data Cleaning & EDA
  │
  ▼
Data Preprocessing
  │
  ├── Numerical Feature Processing
  └── Categorical Feature Processing
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
Pemilihan Model Terbaik
  │
  ▼
model.pkl + preprocessor.pkl
  │
  ▼
FastAPI
  │
  ▼
Endpoint Prediction
  │
  ▼
Hasil Prediksi Churn
```

---

## Dataset

Project menggunakan dataset **Bank Customer Churn Prediction**.

Dataset berisi informasi mengenai karakteristik dan aktivitas nasabah yang digunakan untuk memprediksi status churn.

### Fitur

| Fitur              | Deskripsi                             |
| ------------------ | ------------------------------------- |
| `customer_id`      | ID unik nasabah                       |
| `credit_score`     | Skor kredit nasabah                   |
| `country`          | Negara tempat nasabah berada          |
| `gender`           | Jenis kelamin nasabah                 |
| `age`              | Usia nasabah                          |
| `tenure`           | Lama nasabah menggunakan layanan bank |
| `balance`          | Saldo rekening nasabah                |
| `products_number`  | Jumlah produk bank yang digunakan     |
| `credit_card`      | Kepemilikan kartu kredit              |
| `active_member`    | Status keaktifan nasabah              |
| `estimated_salary` | Estimasi gaji nasabah                 |
| `churn`            | Target yang menunjukkan status churn  |

`customer_id` digunakan sebagai identifier dan tidak digunakan sebagai fitur untuk proses prediksi.

---

## Data Preprocessing

Tahap preprocessing dilakukan untuk mempersiapkan data sebelum digunakan oleh model Machine Learning.

Preprocessing dilakukan menggunakan komponen dari **Scikit-learn**, termasuk `Pipeline` dan `ColumnTransformer`.

### Fitur Numerik

Fitur numerik diproses melalui tahapan preprocessing yang sesuai dengan pipeline training.

Contoh fitur numerik:

```text
credit_score
age
tenure
balance
products_number
estimated_salary
```

### Fitur Kategorikal

Fitur kategorikal diproses menggunakan encoding agar dapat digunakan oleh model Machine Learning.

Contoh fitur kategorikal:

```text
country
gender
credit_card
active_member
```

Pipeline preprocessing yang telah di-fit pada data training disimpan dalam:

```text
src/preprocessor.pkl
```

File tersebut digunakan kembali pada saat inference agar data yang masuk melalui API mendapatkan preprocessing yang konsisten dengan data saat training.

---

## Model Machine Learning

Beberapa algoritma Machine Learning digunakan dan dibandingkan dalam proses pengembangan model:

* Logistic Regression
* Decision Tree / Random Forest
* XGBoost

Evaluasi model dilakukan menggunakan beberapa metrik klasifikasi, yaitu:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Pemilihan model tidak hanya berdasarkan accuracy, tetapi juga mempertimbangkan metrik lain yang relevan terhadap kebutuhan prediksi churn.

Model yang dipilih kemudian disimpan dalam format pickle sebagai:

```text
src/model.pkl
```

Model tersebut digunakan oleh FastAPI untuk melakukan inference terhadap data baru.

---

## Model Artifacts

Project menggunakan dua file utama untuk proses inference:

```text
src/
├── model.pkl
└── preprocessor.pkl
```

### `model.pkl`

Berisi model Machine Learning yang telah dilatih dan digunakan untuk menghasilkan prediksi churn.

### `preprocessor.pkl`

Berisi preprocessing pipeline yang telah di-fit pada data training.

Kedua file tersebut merupakan satu kesatuan dalam proses inference:

```text
Data Input
    │
    ▼
preprocessor.pkl
    │
    ▼
Transformed Data
    │
    ▼
model.pkl
    │
    ▼
Prediction
```

> `model.pkl` dan `preprocessor.pkl` harus berasal dari proses training yang kompatibel. Perubahan terhadap fitur atau preprocessing perlu diikuti dengan proses training dan penyimpanan ulang kedua artifact tersebut.

---

## FastAPI

Model Machine Learning disediakan melalui REST API menggunakan **FastAPI**.

FastAPI bertindak sebagai penghubung antara model Machine Learning dengan aplikasi yang dikembangkan oleh software developer.

Alur integrasinya:

```text
Software / Backend Application
            │
            │ HTTP Request
            ▼
       FastAPI Service
            │
            ▼
    Request Validation
            │
            ▼
    preprocessor.pkl
            │
            ▼
        model.pkl
            │
            ▼
     Prediction Result
            │
            ▼
       JSON Response
```

---

## Struktur Project

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

### Penjelasan Folder

| Folder/File        | Fungsi                                                           |
| ------------------ | ---------------------------------------------------------------- |
| `api/`             | Berisi kode FastAPI untuk menyediakan layanan prediksi           |
| `data/`            | Berisi dataset yang digunakan dalam pengembangan model           |
| `notebooks/`       | Berisi notebook dan script untuk EDA serta workflow Data Science |
| `src/`             | Berisi kode preprocessing, training, serta model artifacts       |
| `model.pkl`        | Model Machine Learning hasil training                            |
| `preprocessor.pkl` | Preprocessing pipeline hasil training                            |
| `README.md`        | Dokumentasi project                                              |
| `requirements.txt` | Daftar dependency Python yang digunakan                          |

---

## Instalasi

Clone repository:

```bash
git clone https://github.com/Boekanadip/bank-churn-prediction.git
cd bank-churn-prediction
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment pada Windows:

```bash
.venv\Scripts\activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

---

## Menjalankan FastAPI

Pastikan file berikut tersedia:

```text
src/model.pkl
src/preprocessor.pkl
```

Kemudian jalankan FastAPI:

```bash
uvicorn api.main:app --reload
```

Setelah berhasil dijalankan, API dapat diakses melalui:

```text
http://127.0.0.1:8000
```

Dokumentasi interaktif FastAPI tersedia pada:

```text
http://127.0.0.1:8000/docs
```

Swagger UI pada `/docs` dapat digunakan untuk mencoba endpoint prediction secara langsung.

---

## API Endpoint

### Health Check

```http
GET /health
```

Endpoint ini digunakan untuk memastikan bahwa service API berjalan dan model berhasil dimuat.

Contoh response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### Prediksi Churn

```http
POST /predict
```

Endpoint ini menerima data nasabah dan mengembalikan hasil prediksi churn.

Contoh request:

```json
{
  "customer_id": "a1b2c3d4",
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

Contoh response:

```json
{
  "customer_id": "a1b2c3d4",
  "churn_probability": 0.7345,
  "churn_percentage": 73,
  "risk_level": "Merah",
  "model_version": "v1.0-xgboost"
}
```

Nilai response di atas merupakan **contoh format response**, bukan hasil prediksi aktual.

---

## Klasifikasi Risiko

Probabilitas churn kemudian dapat dikelompokkan menjadi tiga tingkat risiko:

| Probabilitas Churn | Tingkat Risiko |
| ------------------ | -------------- |
| `< 0.30`           | Hijau          |
| `0.30 – < 0.70`    | Kuning         |
| `>= 0.70`          | Merah          |

Interpretasi:

* **Hijau** → probabilitas churn relatif rendah.
* **Kuning** → probabilitas churn berada pada tingkat menengah.
* **Merah** → probabilitas churn relatif tinggi.

Threshold tersebut merupakan aturan bisnis pada layer API dan dapat disesuaikan apabila terdapat kebutuhan bisnis yang berbeda.

---

## Peran dalam Pengembangan Sistem

Project ini dikembangkan dengan pemisahan tanggung jawab antara Data Science dan Software Development.

### Data Science

Bagian Data Science mencakup:

* Data understanding
* Data cleaning
* Exploratory Data Analysis
* Data preprocessing
* Feature preparation
* Model training
* Model comparison
* Model evaluation
* Pemilihan model terbaik
* Penyimpanan model dalam bentuk `model.pkl`
* Penyimpanan preprocessing pipeline dalam bentuk `preprocessor.pkl`
* Penyediaan model melalui FastAPI
* Dokumentasi kebutuhan integrasi model

### Software Development

Software developer menggunakan endpoint FastAPI untuk mengintegrasikan model Machine Learning ke dalam aplikasi atau sistem yang dikembangkan.

Dengan pendekatan ini, model Machine Learning tidak perlu ditanamkan langsung ke dalam aplikasi utama. Aplikasi cukup mengirimkan data melalui API dan menerima hasil prediksi dalam format JSON.

---

## Alur Integrasi dengan Software

```text
┌───────────────────────────┐
│    Software Application   │
│       / Backend           │
└─────────────┬─────────────┘
              │
              │ POST /predict
              ▼
┌───────────────────────────┐
│         FastAPI           │
│       ML Service          │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    preprocessor.pkl       │
│     Data Transformation   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│        model.pkl          │
│    Churn Classification   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      JSON Response        │
│ Probability + Risk Level  │
└───────────────────────────┘
```

---

## Catatan Pengembangan

Beberapa hal perlu diperhatikan ketika melakukan perubahan terhadap model:

1. Perubahan fitur input dapat memerlukan perubahan pada preprocessing pipeline.
2. Perubahan preprocessing harus diikuti dengan training ulang model.
3. `model.pkl` dan `preprocessor.pkl` harus dibuat dari pipeline training yang kompatibel.
4. Jika model diganti, `model_version` pada API perlu diperbarui.
5. Struktur request API harus tetap konsisten dengan fitur yang digunakan model.
6. Dataset produksi yang mengandung informasi sensitif tidak boleh dimasukkan ke repository publik.
7. Model artifact yang digunakan pada production sebaiknya memiliki versioning yang jelas.

---

## Teknologi yang Digunakan

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* FastAPI
* Uvicorn
* Joblib

---

## Repository

Repository project:

`https://github.com/Boekanadip/bank-churn-prediction`

---

## Status Project

**Status:** Model Machine Learning dan REST API telah disiapkan untuk integrasi dengan software application.

Project ini merupakan implementasi workflow Data Science yang menghubungkan proses pengembangan model Machine Learning dengan kebutuhan integrasi Software Development.
