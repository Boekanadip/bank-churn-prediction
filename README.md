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


> **Status skema data:** Project ini menggunakan dataset
> [mathchi/churn-for-bank-customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
> (Kaggle). Kolom asli menggunakan PascalCase (`CustomerId`, `CreditScore`,
> `Geography`, dst) — lihat bagian **API Contract** di bawah untuk detail
> penamaan field yang dipakai di request/response.


## Struktur Project

```
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
└── lesson_learned_template.md
```

## Quickstart

Ringkas — detail lengkap ada di [`setup_environment.md`](./setup_environment.md).

```bash
# 1. Buat & aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependency (versi sudah dikunci di requirements.txt)
pip install -r requirements.txt

# 3. Jalankan preprocessing (hasil tersimpan otomatis ke src/)
python src/preprocessing.py

# 4. Jalankan training (hasil tersimpan otomatis ke src/)
python src/train.py

# 5. Jalankan API
uvicorn api.main:app --reload
```

Buka `http://127.0.0.1:8000/docs` untuk mencoba endpoint lewat Swagger UI.

## API Contract

### `POST /predict`

**Request body** — field mengikuti penamaan kolom asli dataset (PascalCase):

```json
{
  "CustomerId": "15634602",
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 5,
  "Balance": 125000.50,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 78000.00
}
```

> **Tidak ada field `Surname`/nama nasabah.** Model ML tidak menerima atau
> memproses data pribadi nasabah. Kalau dashboard perlu menampilkan nama
> bersama skor risiko, itu digabungkan (`JOIN`) di sisi Laravel/React
> berdasarkan `CustomerId`, dari data yang sudah tersimpan di database
> Core App — bukan dikirim ke API ML ini.

**Error responses:**

| Status | Kapan terjadi |
|---|---|
| `422 Unprocessable Entity` | Field request salah tipe, hilang, atau di luar rentang valid (lihat `feature_valid_range` di `src/metadata.json`) |
| `500 Internal Server Error` | Kegagalan internal saat memproses model — cek log server, biasanya karena mismatch versi library atau file `.pkl` tidak konsisten (lihat `lesson_learned_template.md`) |

### `GET /health`

Cek apakah API dan model sudah siap.
```json
{ "status": "ok", "model_loaded": true }
```

## Fitur yang Dipakai Model

| Kolom asli (CSV) | Dipakai sebagai fitur? | Keterangan |
|---|---|---|
| `RowNumber` | ❌ | Index baris, dibuang saat load data |
| `CustomerId` | ❌ (identifier) | Dipakai untuk referensi response, bukan fitur model |
| `Surname` | ❌ | PII, tidak pernah masuk pipeline ML |
| `CreditScore` | ✅ | Numerik |
| `Geography` | ✅ | Kategorikal |
| `Gender` | ✅ | Kategorikal |
| `Age` | ✅ | Numerik |
| `Tenure` | ✅ | Numerik |
| `Balance` | ✅ | Numerik |
| `NumOfProducts` | ✅ | Kategorikal |
| `HasCrCard` | ✅ | Kategorikal |
| `IsActiveMember` | ✅ | Kategorikal |
| `EstimatedSalary` | ✅ | Numerik |
| `Exited` | 🎯 Target | Label yang diprediksi (`churn`) |

Semua 10 fitur (bukan hanya subset) dipertahankan berdasarkan hasil analisis
feature importance & korelasi — lihat catatan di `notebooks/churn_predict.py`.

## Dokumen Terkait

- [`setup_environment.md`](./setup_environment.md) — cara setup environment ML dari nol
- [`lesson_learned_template.md`](./lesson_learned_template.md) — kendala teknis yang pernah dialami & solusinya
- `src/metadata.json` — versi library & metrik model aktif (digenerate otomatis oleh `train.py`)

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

