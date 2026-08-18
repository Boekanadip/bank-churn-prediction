"""
api/main.py

Model Serving — FastAPI
Endpoint REST yang menerima input profil nasabah dan mengembalikan
"Probabilitas Churn" (0-100%), sesuai API_Contract_Churn_Prediction.md.

Cara jalanin:
    uvicorn main:app --reload --port 8000

Dokumentasi otomatis (Swagger UI): http://localhost:8000/docs
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =========================================================
# 1. Load model & preprocessor SEKALI saja saat server start
#    (bukan setiap ada request — supaya API responsif/cepat)
# =========================================================
MODEL_PATH = "src/model.pkl"
PREPROCESSOR_PATH = "src/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Endpoint untuk memprediksi probabilitas churn nasabah bank.",
    version="1.0.0",
)


# =========================================================
# 2. Skema Request & Response
#    Pydantic otomatis melakukan validasi tipe data —
#    kalau tim Laravel kirim data yang salah tipe/field kurang,
#    otomatis dapat error 422 dengan pesan yang jelas.
# =========================================================

class CustomerProfile(BaseModel):
    customer_id: str = Field(..., description="UUID nasabah dari database")
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "a1b2c3d4-e5f6-47a8-9b12-cd34ef567890",
                "credit_score": 650,
                "country": "France",
                "gender": "Female",
                "age": 42,
                "tenure": 5,
                "balance": 125000.50,
                "products_number": 2,
                "credit_card": 1,
                "active_member": 1,
                "estimated_salary": 78000.00,
            }
        }


class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_percentage: int
    risk_level: str #belum fix kepake
    model_version: str


# =========================================================
# 3. Fungsi Penentu Risk Level
#    Threshold ditentukan di sisi ML (bukan di Laravel/Frontend)
#    supaya logic-nya konsisten di satu tempat.
#    Sesuai kesepakatan di API_Contract_Churn_Prediction.md
# =========================================================

def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Hijau"
    elif probability < 0.70:
        return "Kuning"
    else:
        return "Merah"


# =========================================================
# 4. Endpoint Utama
# =========================================================

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerProfile):
    # Ubah request jadi DataFrame satu baris, dengan urutan kolom
    # yang sama persis seperti waktu training.
    input_df = pd.DataFrame([{
        "credit_score": customer.credit_score,
        "country": customer.country,
        "gender": customer.gender,
        "age": customer.age,
        "tenure": customer.tenure,
        "balance": customer.balance,
        "products_number": customer.products_number,
        "credit_card": customer.credit_card,
        "active_member": customer.active_member,
        "estimated_salary": customer.estimated_salary,
    }])

    # Preprocessing pakai pipeline yang SAMA seperti waktu training
    # (preprocessor.pkl) — ini kuncinya supaya hasil prediksi konsisten.
    processed_input = preprocessor.transform(input_df)

    # predict_proba mengembalikan [prob_kelas_0, prob_kelas_1]
    # kita ambil index 1 = probabilitas churn
    churn_probability = float(model.predict_proba(processed_input)[0][1])

    return ChurnPredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(churn_probability, 4),
        churn_percentage=round(churn_probability * 100),
        risk_level=get_risk_level(churn_probability),
        model_version="v1.0-xgboost",
    )


# =========================================================
# 5. Endpoint Tambahan — Health Check
#    Berguna untuk tim Laravel/DevOps cek apakah API ML masih hidup,
#    sebelum troubleshoot hal lain yang lebih rumit.
# =========================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


# =========================================================
# 6. Error Handler — Validasi Gagal
#    Supaya pesan error yang dikirim ke tim Laravel jelas dan konsisten
#    dengan format di API_Contract_Churn_Prediction.md
# =========================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Model gagal memproses request. Silakan coba lagi atau hubungi tim DS.",
        },
    )
