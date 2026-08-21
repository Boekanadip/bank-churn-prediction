# Setup Environment — ML Engine

**Project:** Bank Customer Churn Prediction System

**Bagian:** Machine Learning Engine (Data Science)

**Tujuan dokumen:** Panduan agar siapa pun (termasuk anggota tim baru saat onboarding) bisa menjalankan ML engine dari nol, tanpa harus tanya langsung ke orang yang bikin.

---

## 0. Alur yang Direkomendasikan (Update Berdasarkan Pengalaman Sprint)

Berdasarkan pengalaman nyata selama sprint berjalan, ada 2 kemungkinan environment yang bisa dipakai — **keduanya sah dipakai**, tergantung tahap kerjaannya:

| Tahap | Environment yang Direkomendasikan | Alasan |
|---|---|---|
| EDA, Data Preprocessing, Model Training & Evaluation | **Google Colab** | Tidak perlu install apapun di lokal, library umum (pandas, sklearn, dll) sudah tersedia default, lebih ringan untuk laptop dengan spek terbatas |
| Model Serving (FastAPI) | **Lokal (VS Code)** | FastAPI adalah server yang harus terus menyala menunggu request — tidak cocok dijalankan di dalam cell Colab (Colab akan macet menunggu server berhenti, dan sesi bisa terputus sewaktu-waktu) |

**Alur yang disarankan:**
1. Kerjakan EDA → Preprocessing → Training di Colab (notebook `.py` yang sudah disiapkan tim DS)
2. Download hasil `model.pkl` dan `preprocessor.pkl` dari Colab (`files.download(...)`)
3. Pindahkan kedua file itu ke folder `src/` di project lokal
4. Lanjutkan setup lokal dari **Step 1** di bawah, khusus untuk menjalankan `api/main.py`

> Catatan: kalau laptop tim cukup kuat dan ingin semua tahap (termasuk EDA) dikerjakan lokal dari awal, itu juga tetap valid — langkah di bawah ini tetap berlaku untuk skenario itu, cukup lewati bagian download dari Colab.

---

## 1. Prasyarat

| Tool | Versi Minimal | Cara Cek |
|---|---|---|
| Python | 3.10+ | `python --version` |
| pip | terbaru | `pip --version` |
| Git | any | `git --version` |

Cek versi:
- python --version
- pip --version
- git --version
---

## 2. Langkah Setup dari Nol

### Step 1 — Clone Repository

```bash
git clone https://github.com/Boekanadip/bank-churn-prediction
cd bank-churn-prediction
```

### Step 2 — Buat Virtual Environment

Wajib pakai virtual environment supaya library project ini nggak bentrok dengan project Python lain.

```bash
# Buat environment
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Mac/Linux)
source venv/bin/activate
```

> Tandanya berhasil: muncul `(venv)` di depan baris terminal.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Isi `requirements.txt` minimal:

```
fastapi
uvicorn[standard]
pandas
numpy==2.2.6
xgboost==3.4.0
scikit-learn==1.8.0
joblib==1.5.3
matplotlib
seaborn
```

### Step 4 — Siapkan Dataset & Model

Ada 2 opsi, pilih sesuai alur di bagian 0:

**Opsi A — Sudah punya `model.pkl` & `preprocessor.pkl` dari Colab:**
- Taruh langsung kedua file itu di folder `src/`
- Lewati Step 5, langsung lanjut ke Step 6 (Jalankan API Server)

**Opsi B — Mau reproduce dari awal secara lokal:**
- Download dataset dari [https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset]
- Taruh file di folder `data`
- **Jangan commit file dataset mentah ke Git** kalau ukurannya besar (sudah di-exclude lewat `.gitignore`)
- Lanjut ke Step 5

### Step 5 — Jalankan EDA & Training (hanya untuk Opsi B)

```bash
python src/preprocessing.py
python src/train.py
```

Output yang dihasilkan: file model (`model.pkl`) di folder `src/`.

> XGBoost merupakan dependency wajib karena model production menggunakan XGBoost. Jika XGBoost gagal di-install atau terjadi version mismatch, proses setup harus diperbaiki sebelum menjalankan model serving.

### Step 6 — Jalankan API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Setelah jalan, cek:
- API aktif di: `http://localhost:8000`
- Dokumentasi otomatis (Swagger UI): `http://localhost:8000/docs` — bisa dipakai untuk test endpoint langsung dari browser tanpa Postman


## 3. Masalah Umum & Solusi (FAQ Setup)

> Bagian ini berisikan masalah saat setup, tambahkan di sini supaya orang berikutnya tidak stuck di masalah yang sama.

| Masalah | Kemungkinan Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError` | Virtual environment belum diaktifkan, atau `pip install` belum dijalankan | Pastikan `(venv)` muncul di terminal, lalu ulangi `pip install -r requirements.txt` |
| Port 8000 sudah dipakai | Ada proses lain yang jalan di port yang sama | Jalankan dengan port lain: `uvicorn api.main:app --reload --port 8001` |
| VS Code sangat lambat / hang saat run cell pertama kali | RAM terpakai tinggi (misal >80%) dan/atau antivirus melakukan real-time scan pada folder project setiap file diakses | 1) Exclude folder project dari antivirus (Windows Security → Virus & threat protection → Manage settings → Add or remove exclusions), 2) tutup aplikasi lain yang tidak perlu, 3) restart VS Code. Kalau masih lambat, pertimbangkan kerjakan tahap EDA/Preprocessing/Training di Google Colab dulu (lihat bagian 0), baru pindah ke lokal khusus untuk API |
| Error JavaScript di Google Colab (\"Could not load the JavaScript files...\") | Sesi login Google expired, atau browser blokir third-party cookies | Reload halaman, login ulang akun Google, pastikan third-party cookies diizinkan untuk domain google.com, hindari mode Incognito |


## 4. Struktur Folder ML Engine (Sementara)
bank-churn-prediction/
├── data/
│   Churn.csv
├── notebooks/
│   └── churn_predict.py # code mulai dari persiapan-penyimpanan model
├── src/
│   ├── preprocessing.py  # Pipeline cleaning, encoding, scaling
│   ├── preprocessor.pkl  # Untuk Pembersihan inputan 
│   ├── train.py          # Script
│   ├── metadata.json     # Catatan versi
│   └── model.pkl         # Model hasil training (bukan source code, hasil biner)
├── api/
│   └── main.py           # FastAPI app, endpoint /predict
├── requirements.txt
└── README.md              
├── setup_environment_ml.md     
└── lesson_learned_template.md

```

## 5. Best practices

- Kunci versi dependency di `requirements.txt`.
- Simpan metadata training di `src/metadata.json` (isi: versi library, tanggal training, model_version).
- Untuk reproducibility, pertimbangkan Dockerfile yang menjalankan environment yang sama untuk training & serving.
- Jangan commit dataset mentah ke repo publik.

---

## 6. Kontak / Penanggung Jawab

| Bagian | Nama | Kontak |
|---|---|---|
| ML Engine & Rest API | Adib Raihan A. | Email: adib.raihann@gmail.com |
