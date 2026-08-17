# Lesson Learned — Bank Customer Churn Prediction System

**Tujuan dokumen:** Mencatat kendala nyata yang dialami selama project (khususnya soal integrasi antar tim ML dan Software Dev), beserta solusinya. Dokumen ini akan jadi basis Knowledge Management System (KMS) untuk onboarding tim di masa depan — jadi tulis dengan asumsi pembacanya adalah orang baru yang belum tahu apa-apa soal project ini.

> **Cara pakai:** Isi tabel di bawah **selama sprint berjalan**, begitu masalah terjadi dan solusinya ketemu — jangan ditunda sampai akhir project, karena detailnya bisa lupa.

---

## 1. Log Kendala & Solusi

| Tanggal | Kategori | Masalah yang Dialami | Penyebab | Solusi | Dicatat oleh |
|---|---|---|---|---|---|
| _contoh:_ 15 Agu | Integrasi API | Tim Laravel dapat error 422 saat call endpoint `/predict` | Format `active_member` yang dikirim Laravel berupa boolean (`true`/`false`), sementara FastAPI expect integer (`0`/`1`) | Disepakati ulang: FastAPI terima keduanya via validasi Pydantic yang lebih fleksibel | _(nama)_ |
| 17 Agu | Setup Environment | VS Code sangat lambat / hang saat menjalankan cell pertama kali (proses baca file dataset & start kernel Jupyter tidak kunjung selesai) | Kombinasi 2 faktor: (1) RAM laptop terpakai ~80% oleh aplikasi lain, (2) antivirus/antimalware melakukan real-time scan pada folder project setiap kali file diakses, memperlambat proses baca file dan start kernel | Tambahkan folder project ke exclusion list antivirus (Windows Security → Virus & threat protection → Manage settings → Add or remove exclusions → Add folder project), tutup aplikasi lain yang tidak perlu untuk melonggarkan RAM, lalu restart VS Code | _(nama)_ |
| | Data/Model | | | | |
| | Database (UUID) | | | | |
| | Deployment | | | | |
| | Lainnya | | | | |

**Kategori yang bisa dipakai:** Setup Environment, Integrasi API, Data/Model, Database (UUID), Autentikasi, Deployment, Komunikasi Tim, Lainnya.

---

## 2. Kendala Spesifik Integrasi ML ↔ Backend

Bagian ini fokus ke masalah yang paling sering terjadi di project seperti ini — integrasi antara ML Engine (FastAPI) dan Core App (Laravel). Isi berdasarkan pengalaman nyata tim.

### 2.1 Ketidakcocokan Format Data
_Contoh yang perlu diisi: field apa yang sempat beda format, gimana cara nemuinnya, gimana solusinya (misal: sepakati ulang API contract, tambah validasi, dll)_

### 2.2 Masalah UUID
_Contoh yang perlu diisi: apakah ada masalah waktu passing UUID antara Laravel dan FastAPI (misal: format UUID versi berapa, ada kesalahan tipe data, dll)_

### 2.3 Response Time / Performance
_Contoh yang perlu diisi: apakah API ML sempat lambat, apa penyebabnya (misal model terlalu besar, tidak ada caching), dan solusinya_

### 2.4 Error Handling
_Contoh yang perlu diisi: skenario apa yang bikin API error tak terduga, dan bagaimana akhirnya ditangani di kedua sisi_

---

## 3. Apa yang Akan Dilakukan Berbeda Kalau Mengulang Project Ini

Bagian reflektif — tulis di akhir sprint, berdasarkan pengalaman keseluruhan tim.

- **Kalau bisa mulai dari awal lagi, apa yang akan disiapkan lebih dulu?**
  _(contoh: sepakati API contract lebih detail sebelum coding dimulai, bukan sambil jalan)_

- **Proses mana yang paling makan waktu tapi sebenarnya bisa lebih cepat?**
  _(isi)_

- **Tools/library apa yang ternyata sangat membantu, dan mana yang sebaiknya dihindari?**
  _(isi)_

---

## 4. Rekomendasi untuk Tim Berikutnya (Onboarding)

Bagian ini yang paling penting untuk KMS — tulis seolah menjelaskan ke orang yang benar-benar baru gabung project sejenis.

- **3 hal yang harus dilakukan di awal sprint sebelum mulai coding:**
  1.
  2.
  3.

- **3 kesalahan yang sebaiknya dihindari** (berdasarkan pengalaman project ini):
  1.
  2.
  3.

- **Dokumen/resource yang wajib dibaca duluan** sebelum mulai kerja:
  - `SETUP_ENVIRONMENT_ML.md`
  - `API_Contract_Churn_Prediction.md`
  - _(tambahkan lainnya)_

---

## 5. Kontributor Dokumen

| Nama | Role | Kontribusi |
|---|---|---|
| | | |
