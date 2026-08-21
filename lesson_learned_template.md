# Lesson Learned — Bank Customer Churn Prediction System

**Tujuan dokumen:** Mencatat kendala nyata yang dialami selama project
(khususnya soal integrasi antar tim ML dan Software Dev), beserta
solusinya. Dokumen ini jadi basis Knowledge Management System (KMS) untuk
onboarding tim di masa depan — ditulis dengan asumsi pembacanya adalah
orang baru yang belum tahu apa-apa soal project ini.

> **Cara pakai:** Isi tabel di bawah **selama sprint berjalan**.

---

## 1. Log Kendala & Solusi

| Tanggal | Kategori | Masalah yang Dialami | Penyebab | Solusi | Dicatat oleh |
|---|---|---|---|---|---|
| 17 Agustus | Setup Environment | VS Code sangat lambat / hang saat menjalankan cell pertama kali | Kombinasi (1) RAM laptop terpakai ~80% oleh aplikasi lain, (2) antivirus melakukan real-time scan pada folder project setiap kali file diakses | Tambahkan folder project ke exclusion list antivirus, tutup aplikasi lain untuk melonggarkan RAM, restart VS Code | Adib raihan a. |
| 18 Agustus | Data/Model | Model XGBoost bermasalah saat dipakai di environment dengan versi XGBoost berbeda | Ketidakcocokan versi library antara environment training dan serving | Samakan versi XGBoost antara training dan serving, pin dependency di `requirements.txt` | Adib raihan a. |
| 19 Agustus | Integrasi API | API gagal start saat `uvicorn api.main:app --reload` | `.pkl` disimpan dengan scikit-learn versi berbeda dari environment yang memuatnya. Format pickle scikit-learn tidak dijamin kompatibel antar versi | Samakan versi scikit-learn di environment serving dengan versi saat file `.pkl` dihasilkan (`pip install scikit-learn==<versi-sama>`), verifikasi dengan `pip show scikit-learn` | Adib raihan a. |
| 19 Agustus | Prediksi tidak reliable | Input dengan `balance`/`estimated_salary` bernilai jutaan menghasilkan probabilitas mencurigakan (SHAP menunjukkan nilai ter-scale belasan standar deviasi dari rata-rata) | Input jauh melebihi rentang data training, model melakukan ekstrapolasi ke wilayah yang tidak pernah dipelajari | Tambahkan validasi rentang nilai di skema Pydantic (`Field(ge=..., le=...)`) sehingga input di luar rentang wajar ditolak (HTTP 422) | Adib raihan a. |
| 20 Agustus | Integrasi API | `AttributeError: 'CustomerProfile' object has no attribute 'creditscore'. Did you mean: 'CreditScore'?` saat POST `/predict` | Salah casing saat mengakses atribut Pydantic — field didefinisikan `CreditScore` (PascalCase) tapi diakses sebagai `customer.creditscore` (lowercase). Python bersifat case-sensitive, dua penulisan ini dianggap atribut berbeda | Samakan casing persis: `customer.CreditScore`, bukan `customer.creditscore`. Cek ulang seluruh field lain di blok yang sama untuk pola serupa | Adib raihan a. |
| 20 Agustus | Integrasi API | `pydantic_core.ValidationError: Field required` untuk `customer_id` saat return response, padahal sudah diisi `CustomerId=...` | Field response didefinisikan `customer_id` (snake_case) di class `ChurnPredictionResponse`, tapi saat konstruksi objek dipakai kwarg `CustomerId` (PascalCase) — dianggap dua field berbeda | Samakan casing kwarg saat konstruksi response dengan definisi class: `customer_id=customer.CustomerId` (kiri mengikuti nama field response, kanan mengikuti nama atribut request) | Adib raihan a. |
| | | | | | |
| | | | | | |

**Kategori yang bisa dipakai:** Setup Environment, Integrasi API, Data/Model, Database (UUID), Autentikasi, Deployment, Komunikasi Tim, Lainnya.

---

## 2. Kendala Spesifik Integrasi ML ↔ Backend

### 2.1 Ketidakcocokan Format Data

Terjadi dua bentuk ketidakcocokan format selama project ini:

**a. Ketidakcocokan skema fitur.** Sempat muncul rencana menyederhanakan
endpoint `/predict` agar hanya menerima 4 field, padahal skema
`CustomerProfile`, `input_df` di fungsi `predict_churn()`, dan kolom yang
dipakai `preprocessor.pkl` saat training harus selalu identik. Solusi:
analisis feature importance dulu sebelum ubah skema — hasilnya
menunjukkan fitur seperti `Age` dan `Geography` tetap signifikan, sehingga
skema 10 fitur dipertahankan.

**b. Ketidakcocokan penamaan (casing).** Saat migrasi dataset ke sumber
Kaggle yang kolomnya PascalCase (`CreditScore`, `Geography`, dst), banyak
titik di `main.py` yang masih menulis nama atribut dengan casing salah
(lihat baris log 20 Agustus di atas). Solusi jangka panjang: pertimbangkan
memakai `Field(alias=...)` di Pydantic supaya field internal tetap
snake_case (konsisten, mudah diketik) sementara API tetap menerima JSON
PascalCase dari client.

### 2.2 Masalah UUID
_( pengalaman Software Dev )_

### 2.3 Response Time / Performance
_API ML sempat lambat, penyebabnya tidak ada caching._

### 2.4 Error Handling

`main.py` menggunakan generic exception handler yang mengembalikan format
JSON konsisten (`{"error": "internal_error", "message": "..."}`) untuk
semua error tak terduga (HTTP 500). Ini membantu tim Laravel mendapat
format error yang predictable, tapi juga berarti **pesan error asli
(traceback) tersembunyi dari client** — untuk debugging, penyebab
sebenarnya harus dicek di log terminal `uvicorn`, bukan dari response body.
Ini kenapa isi kolom "Response body" di Swagger UI hanya menampilkan pesan
generik meski error aslinya beragam (lihat baris log 20 Agustus).

---

## 3. Apa yang Akan Dilakukan Berbeda Kalau Mengulang Project Ini

- **Kalau bisa mulai dari awal lagi:**
  1. Sepakati API contract lebih detail sebelum coding dimulai, bukan sambil jalan
  2. Kunci versi seluruh library di `requirements.txt` sejak hari pertama, dan selalu training + serving di environment yang sama
  3. Tentukan konvensi penamaan field (snake_case vs PascalCase) di satu tempat sejak awal, dan konsisten di seluruh kode — bukan mengikuti casing kolom CSV mentah secara langsung, untuk menghindari bug typo casing yang berulang

---

## 4. Rekomendasi untuk Tim Berikutnya (Onboarding)

- **Hal yang harus dilakukan di awal sprint sebelum mulai coding:**
  1. Diskusikan dengan tim Software Developer, API yang akan diterima serta strukturnya
  2. Sepakati dan kunci versi seluruh library (`scikit-learn`, `xgboost`, `joblib`, `numpy`) di `requirements.txt` sebelum mulai coding
  3. Tentukan rentang nilai valid untuk tiap fitur numerik, dan sepakati skala satuan (mata uang, dst) dengan tim Software Dev

- **Kesalahan yang sebaiknya dihindari** (berdasarkan pengalaman project ini):
  1. Mulai dari kode terlebih dahulu ketimbang diskusi dengan tim Software Developer
  2. Tidak menyusun repository dengan baik dan benar
  3. Mencampur konvensi penamaan (snake_case dan PascalCase) tanpa dokumentasi yang jelas tentang field mana yang pakai casing apa — sumber bug `AttributeError`/`ValidationError` yang berulang

- **Dokumen/resource yang wajib dibaca duluan** sebelum mulai kerja:
  - `README.md`
  - `setup_environment.md`
  - `API_Contract_Churn_Prediction.md`

---

## 5. Kontributor Dokumen

| Nama | Role | Kontribusi |
|---|---|---|
| Adib raihan a. | Data Scientist | Pembersihan data hingga API Endpoint |
