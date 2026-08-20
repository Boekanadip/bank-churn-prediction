"""
api/main.py

Model Serving — FastAPI
Endpoint REST yang mengembalikan "Probabilitas Churn" (0-100%), sesuai API_Contract_Churn_Prediction.md.

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
    CustomerId: str = Field(..., description="UUID nasabah dari database")
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float = Field(..., ge=0, le=900000000)
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float = Field(..., ge=0, le=9000000)

    class Config:
        json_schema_extra = {
            "example": {
                "CustomerId":   "a1b2c3d4-e5f6-47a8-9b12-cd34ef567890",
                "Surname": "Ujang",
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 5,
                "Balance": 130000.10,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.00,
            }
        }


class ChurnPredictionResponse(BaseModel):
    CustomerId: str
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
    # yang sama persis seperti waktu training.
    input_df = pd.DataFrame([{
        "CreditScore": customer.CreditScore,
        "Geography": customer.Geography,
        "Gender": customer.Gender,
        "Age": customer.Age,
        "Tenure": customer.Tenure,
        "Balance": customer.Balance,
        "NumOfProducts": customer.NumOfProducts,
        "HasCrCard": customer.HasCrCard,
        "IsActiveMember": customer.IsActiveMember,
        "EstimatedSalary": customer.EstimatedSalary,
    }])

    # Preprocessing pakai pipeline yang SAMA seperti waktu training
    # (preprocessor.pkl) — ini kuncinya supaya hasil prediksi konsisten.
    processed_input = preprocessor.transform(input_df)

    # predict_proba mengembalikan [prob_kelas_0, prob_kelas_1]
    # kita ambil index 1 = probabilitas churn
    churn_probability = float(model.predict_proba(processed_input)[0][1])

    return ChurnPredictionResponse(
        CustomerId=customer.CustomerId,
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
