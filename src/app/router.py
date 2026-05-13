from fastapi import FastAPI, HTTPException, APIRouter
from fastapi import File, UploadFile

from .services import model_info_service, predict_service, analyze_emotion_service
from .schemas import PredictRequest

app_router = APIRouter(tags=['Mental Health Prediction'])


@app_router.get(
    "/model-info",
    summary="Get model metadata",
    description="Returns the name, AUC score, and top 5 most influential features of the currently loaded model.",
)
async def model_info_endpoint():
    return await model_info_service()


@app_router.post(
    "/predict",
    summary="Predict mental health treatment likelihood",
    description="Takes workplace and personal factors as input and returns a prediction, confidence score, and the most influential features that drove the result.",
)
async def predict_endpoint(request: PredictRequest):
    return await predict_service(request)

@app_router.post(
    "/analyze-emotion",
    summary="Detect facial emotion from an image",
    description="Accepts a single-face image (JPEG, PNG, WebP, max 5MB) and returns the detected emotional state with a mental health relevance note. Uses a two-stage CV pipeline: face detection followed by emotion classification.",
)
async def analyze_emotion_endpoint(file: UploadFile = File(...)):
    return await analyze_emotion_service(file)