# Lesson Learned — Bank Customer Churn Prediction System

**Tujuan dokumen:** Mencatat kendala nyata yang dialami selama project (khususnya soal integrasi antar tim ML dan Software Dev), beserta solusinya. Dokumen ini akan jadi basis Knowledge Management System (KMS) untuk onboarding tim di masa depan — jadi tulis dengan asumsi pembacanya adalah orang baru yang belum tahu apa-apa soal project ini.

> **Cara pakai:** Isi tabel di bawah **selama sprint berjalan**.

---

## 1. Log Kendala & Solusi

| Tanggal | Kategori | Masalah yang Dialami | Penyebab | Solusi | Dicatat oleh |
|---|---|---|---|---|---|
| 17 Agustus | Setup Environment | VS Code sangat lambat / hang saat menjalankan cell pertama kali (proses baca file dataset & start kernel Jupyter tidak kunjung selesai) | Kombinasi 2 faktor: (1) RAM laptop terpakai ~80% oleh aplikasi lain, (2) antivirus/antimalware melakukan real-time scan pada folder project setiap kali file diakses, memperlambat proses baca file dan start kernel | Tambahkan folder project ke exclusion list antivirus (Windows Security → Virus & threat protection → Manage settings → Add or remove exclusions → Add folder project), tutup aplikasi lain yang tidak perlu untuk melonggarkan RAM, lalu restart VS Code | Adib raihan a. |
| 18 Agustus | Data/Model | Model XGBoost yang dibuat pada environment tertentu mengalami masalah ketika digunakan pada environment dengan versi XGBoost berbeda | Ketidakcocokan versi library antara environment training dan environment serving | Samakan versi XGBoost antara training dan serving serta pin dependency pada requirements.txt | Adib raihan a. |
| |  | | | | |
| |  | | | | |
| |  | | | | |
| |  | | | | |

**Kategori yang bisa dipakai:** Setup Environment, Integrasi API, Data/Model, Database (UUID), Autentikasi, Deployment, Komunikasi Tim, Lainnya.

---

## 2. Kendala Spesifik Integrasi ML ↔ Backend

Bagian ini fokus ke masalah yang paling sering terjadi di project seperti ini — integrasi antara ML Engine (FastAPI) dan Core App (Laravel). Berdasarkan pengalaman nyata tim.

### 2.1 Ketidakcocokan Format Data

### 2.2 Masalah UUID

### 2.3 Response Time / Performance
_API ML sempat lambat, penyebabnya tidak ada caching._

### 2.4 Error Handling

---

## 3. Apa yang Akan Dilakukan Berbeda Kalau Mengulang Project Ini

Bagian reflektif — tulis di akhir sprint, berdasarkan pengalaman keseluruhan tim.

- **Kalau bisa mulai dari awal lagi, apa yang akan disiapkan lebih dulu?**
  _sepakati API contract lebih detail sebelum coding dimulai, bukan sambil jalan_

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
  - `API_Contract_Churn_Prediction.md`
  - `README.MD`

---

## 5. Kontributor Dokumen

| Nama | Role | Kontribusi |
|---|---|---|
| | | |
