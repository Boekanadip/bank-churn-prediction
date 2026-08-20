# Lesson Learned - Bank Customer Churn Prediction System

**Tujuan dokumen:** Mencatat kendala nyata yang dialami selama project (khususnya soal integrasi antar tim ML dan Software Dev), beserta solusinya. Dokumen ini akan jadi basis Knowledge Management System (KMS) untuk onboarding tim di masa depan - jadi tulis dengan asumsi pembacanya adalah orang baru yang belum tahu apa-apa soal project ini.

> **Cara pakai:** Isi tabel di bawah **selama sprint berjalan**.

---

## 1. Log Kendala & Solusi

| Tanggal | Kategori | Masalah yang Dialami | Penyebab | Solusi | Dicatat oleh |
|---|---|---|---|---|---|
| 17 Agustus | Setup Environment | VS Code sangat lambat / hang saat menjalankan cell pertama kali (proses baca file dataset & start kernel Jupyter tidak kunjung selesai) | Kombinasi 2 faktor: (1) RAM laptop terpakai ~80% oleh aplikasi lain, (2) antivirus/antimalware melakukan real-time scan pada folder project setiap kali file diakses, memperlambat proses baca file dan start kernel | Tambahkan folder project ke exclusion list antivirus (Windows Security → Virus & threat protection → Manage settings → Add or remove exclusions → Add folder project), tutup aplikasi lain yang tidak perlu untuk melonggarkan RAM, lalu restart VS Code | Adib raihan a. |
| 18 Agustus | Data/Model | Model XGBoost yang dibuat pada environment tertentu mengalami masalah ketika digunakan pada environment dengan versi XGBoost berbeda | Ketidakcocokan versi library antara environment training dan environment serving | Samakan versi XGBoost antara training dan serving serta pin dependency pada requirements.txt | Adib raihan a. |
| 19 Agustus | Integrasi API | API gagal start saat menjalankan `uvicorn api.main:app --reload` | File .pkl disimpan (joblib.dump) memakai satu versi scikit-learn (misal 1.8.0), sedangkan environment yang memuatnya (joblib.load) memakai versi berbeda (misal 1.6.1). Format pickle internal scikit-learn tidak dijamin kompatibel antar versi - baik ke versi lebih lama maupun lebih baru | Samakan versi scikit-learn di environment serving dengan versi yang tercatat saat file .pkl dihasilkan: pip install scikit-learn==<versi-yang-sama> Verifikasi dengan pip show scikit-learn sebelum menjalankan ulang uvicorn. | adib |
| 19 Agustus | Prediksi tidak reliable | Input test dengan balance dan estimated_salary bernilai jutaan menghasilkan probabilitas churn yang mencurigakan (skor SHAP menunjukkan nilai fitur ter-scale sangat ekstrem, belasan standar deviasi dari rata-rata) | Input test yang dikirim jauh melebihi rentang ini (misal balance = 1.250.000, 5x lipat dari maksimum training). Model melakukan ekstrapolasi ke wilayah data yang belum pernah dipelajari, sehingga hasil prediksi tidak bisa dipercaya meskipun API tidak mengembalikan error. | Tambahkan validasi rentang nilai di skema Pydantic CustomerProfile, contoh: `balance: float = Field(..., ge=0, le=300000)` `estimated_salary: float = Field(..., ge=0, le=250000)` Sehingga input di luar rentang wajar langsung ditolak (HTTP 422) alih alih menghasilkan prediksi yang menyesatkan. | adib |
| |  | | | | |
| |  | | | | |

**Kategori yang bisa dipakai:** Setup Environment, Integrasi API, Data/Model, Database (UUID), Autentikasi, Deployment, Komunikasi Tim, Lainnya.

---

## 2. Kendala Spesifik Integrasi ML ↔ Backend
<a name="customer_id"></a>
Bagian ini fokus ke masalah yang paling sering terjadi di project seperti ini — integrasi antara ML Engine (FastAPI) dan Core App (Laravel). Berdasarkan pengalaman nyata tim.

### 2.1 Ketidakcocokan Format Data
Sempat muncul rencana menyederhanakan endpoint /predict agar hanya menerima 4 field (balance, tenure, products_number, active_member), padahal skema CustomerProfile di main.py, input_df di fungsi predict_churn(), dan kolom yang dipakai preprocessor.pkl saat training harus selalu identik satu sama lain. Kalau salah satu diubah tanpa mengubah dua lainnya, request akan gagal (HTTP 422) atau preprocessor.transform() error. Solusi: analisis feature importance dulu sebelum ubah skema fitur — hasilnya menunjukkan age dan country tetap signifikan, jadi skema 10 fitur dipertahankan.
---

## 3. Apa yang Akan Dilakukan Berbeda Kalau Mengulang Project Ini

Bagian reflektif — tulis di akhir sprint, berdasarkan pengalaman keseluruhan tim.

- **Kalau bisa mulai dari awal lagi, apa yang akan disiapkan lebih dulu?**
  _kunci versi library di requirements.txt sejak hari pertama (pakai ==, bukan versi bebas), dan selalu training + serving di environment yang sama — jangan campur Colab dan lokal tanpa menyamakan versi dulu. Ini kendala yang paling sering muncul berulang di project ini. Serta sepakati API contract lebih detail sebelum coding dimulai, bukan sambil jalan_

---

## 4. Rekomendasi untuk Tim Berikutnya (Onboarding)

- **Hal yang harus dilakukan di awal sprint sebelum mulai coding:**
  1. Diskusikan dengan tim Software Developer, API yang akan diterima serta strukturnya.
  2.
  3.

- **Kesalahan yang sebaiknya dihindari** (berdasarkan pengalaman project ini):
  1. Mulai dari kode terlebih dahulu ketimbang, diskusi dengan tim Software Developer.
  2. Tidak Menyusun repository dengan baik dan benar.
  3.

- **Dokumen/resource yang wajib dibaca duluan** sebelum mulai kerja:
  - `SETUP_ENVIRONMENT_ML.md`
  - `README.MD`

---

## 5. Kontributor Dokumen

| Nama | Role | Kontribusi |
|---|---|---|
| Adib raihan a. | Data Scientist | Pembersihan data hingga API End Point |
