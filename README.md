# Bank Customer Churn Prediction

Project Machine Learning untuk memprediksi kemungkinan **customer churn** pada nasabah bank berdasarkan karakteristik demografi, kondisi finansial, dan aktivitas nasabah.

Dokumentasi ini menjelaskan alur, dataset, cara menjalankan service FastAPI, dan artefak model yang diperlukan untuk inference.

---

## Tujuan

Tujuan project ini:

- Memprediksi apakah seorang nasabah berpotensi melakukan churn.
- Menghasilkan probabilitas churn.
- Mengelompokkan tingkat risiko churn berdasarkan probabilitas prediksi.
- Menyediakan layanan prediksi melalui REST API (FastAPI) agar dapat diintegrasikan dengan aplikasi lain.

Model yang dilatih disajikan oleh FastAPI sebagai layanan terpisah sehingga tim Software Development dapat mengintegrasikannya tanpa menanamkan model ke aplikasi utama.

---

## Ringkasan Alur

Data → Data understanding → Cleaning & EDA → Preprocessing → Train/Test split → Model training → Evaluasi → Pilih model → Simpan artefak (model.pkl + preprocessor.pkl) → FastAPI → Endpoint /predict

---

## Dataset

Project menggunakan dataset "Bank Customer Churn Prediction" https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset. Dataset memuat fitur demografi, saldo, aktivitas, dan flag target `churn`.

Contoh fitur:

- `CustomerId` (ID unik, tidak dipakai sebagai fitur model)
- `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- `Geography`, `Gender`, `HasCrCard`, `IsActiveMember`
- `Exited` (target)

Sumber dataset (contoh): https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

Catatan: Jangan commit dataset mentah ke repository publik. Gunakan .gitignore untuk mengecualikannya.

---

## Instalasi (singkat)

1. Clone repository:

   git clone https://github.com/Boekanadip/bank-churn-prediction.git
   cd bank-churn-prediction

2. Buat virtual environment dan aktifkan:

   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

3. Install dependency:

   pip install -r requirements.txt

Catatan: Disarankan mengunci versi pada requirements.txt (pakai `==`) untuk reproducibility.

---

## Menjalankan API (FastAPI)

Pastikan artefak model tersedia di `src/`:

- `src/model.pkl`
- `src/preprocessor.pkl`

Jalankan server:

   uvicorn api.main:app --reload --port 8000

Buka:

- http://127.0.0.1:8000 — root
- http://127.0.0.1:8000/docs — Swagger UI untuk mencoba endpoint

---

## Artefak Model

- `src/model.pkl` — model terlatih (pickle/joblib)
- `src/preprocessor.pkl` — pipeline preprocessing (fit pada data training)

Keduanya harus kompatibel: jika ada perubahan pada fitur atau preprocessing, lakukan training ulang dan simpan artefak baru.

---

## Metode & Evaluasi

Model yang dieksperimenkan antara lain Logistic Regression, Random Forest, dan XGBoost. Evaluasi menggunakan metrik: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Pemilihan model mempertimbangkan lebih dari satu metrik sesuai kebutuhan bisnis.

---

## Struktur Project (ringkas)

bank-churn-prediction/
├── api/
│   └── main.py                  # FastAPI serving endpoint
├── data/
│   └── Churn_Modelling.csv      # dataset mentah
├── notebooks/
│   └── churn_predict.py         # EDA (Exploratory Data Analysis)
├── src/
│   ├── preprocessing.py         # pipeline preprocessing (fit & save)
│   ├── train.py                 # training & evaluasi model
│   ├── preprocessor.pkl         # (hasil generate, jangan commit manual)
│   ├── model.pkl                # (hasil generate, jangan commit manual)
│   └── metadata.json            # (hasil generate) versi & metrik model aktif
├── requirements.txt
├── README.md                    # dokumen ini
├── setup_environment.md
└── lesson_learned_template.mdd

---

## Troubleshooting singkat

- ModuleNotFoundError: pastikan virtualenv aktif dan `pip install -r requirements.txt` sudah dijalankan.
- Port 8000 terpakai: jalankan uvicorn dengan `--port` lain.
- Error pickle load: pastikan versi scikit-learn/xgboost kompatibel dengan versi yang digunakan saat serialisasi.

---

## Kontribusi

Untuk kontribusi: buka issue atau buat pull request. Sertakan deskripsi perubahan, cara menjalankan, dan testing minimal.

---

## Lisensi

Tambahkan file LICENSE jika diperlukan. Saat ini tidak ada lisensi spesifik pada repo.
