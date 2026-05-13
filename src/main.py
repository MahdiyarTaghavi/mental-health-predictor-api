from fastapi import FastAPI, HTTPException

from src.app.router import app_router
app = FastAPI(
    title="Mental Health in Tech — Predictor API",
    description="Predicts whether a tech worker is likely to seek mental health treatment based on workplace and personal factors.",
    version="1.0.0",
)

app.include_router(app_router)

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Mental Health Prediction is up and running",
        "version": "1.0",
        "status": "healthy"
    }