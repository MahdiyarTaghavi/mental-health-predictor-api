from fastapi import HTTPException
from fastapi import File, UploadFile

from core.ml.model import predict, model_name, auc, feature_importance
from core.ml.vision import analyze_emotion
from app.schemas import PredictRequest
from utils.loggings.logger import app_log as log

async def model_info_service():
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]
    return {
        "model_name": model_name,
        "auc_score": round(auc, 4),
        "top_5_features": [f for f, _ in top_features],
        "description": (
            "Trained on the OSMI Mental Health in Tech Survey dataset. "
            "Predicts likelihood of seeking mental health treatment."
        ),
    }

async def predict_service(request: PredictRequest):
    try:
        result = predict(request.model_dump())
        return result
    except Exception as e:
        log.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong during prediction. Try again later.")

async def analyze_emotion_service(file: UploadFile = File(...)):
    """
    Accepts a facial image and returns detected emotional state.
    Provides a mental health relevance note based on the dominant emotion.
    Supported formats: JPEG, PNG, WebP.
    """
    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 5MB.")

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPEG, PNG, or WebP."
        )

    try:
        result = analyze_emotion(image_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong during prediction. Try again later.")