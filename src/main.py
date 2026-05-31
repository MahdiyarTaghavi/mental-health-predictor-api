from fastapi import FastAPI, HTTPException

from app.router import app_router
from config import settings
from core.middlewares.logging_context import log_request_context_middleware
from core.middlewares.rate_limiter import RedisRateLimitMiddleware
app = FastAPI(
    title="Mental Health in Tech — Predictor API",
    description="Predicts whether a tech worker is likely to seek mental health treatment based on workplace and personal factors.",
    version="1.0.0",
)

app.middleware("http")(log_request_context_middleware)

app.add_middleware(
    RedisRateLimitMiddleware,
    redis_host=settings.REDIS_HOST,
    redis_port=settings.REDIS_PORT,
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW,
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