from deepface import DeepFace
from PIL import Image
import numpy as np
import io

RELEVANCE_MAP = {
    "happy": "Low stress indicators detected.",
    "neutral": "No strong emotional signals detected.",
    "sad": "Low mood indicators detected. May benefit from support.",
    "fear": "Anxiety indicators detected. Professional support recommended.",
    "angry": "High stress indicators detected. May benefit from support.",
    "disgust": "Negative affect detected. May benefit from support.",
    "surprise": "Heightened arousal detected. Context-dependent.",
}

def analyze_emotion(image_bytes: bytes) -> dict:
    """
    Accepts raw image bytes, detects the face, and classifies emotional state.
    Uses a two-stage CV pipeline internally: face detection → emotion classification.
    enforce_detection=False prevents crashes on non-ideal face angles or crops.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    result = DeepFace.analyze(
        img_array,
        actions=["emotion"],
        enforce_detection=False,
        silent=True,
    )

    if len(result) > 1:
        raise ValueError(f"{len(result)} faces detected. Please upload an image with a single face.")

    if result[0].get("face_confidence", 1) < 0.5:
        raise ValueError("No face detected in the image. Please upload a clear photo with a visible face.")

    raw_emotions = result[0]["emotion"]

    emotions_sorted = sorted(
        [
            {"emotion": k, "confidence": round(float(v), 1)}
            for k, v in raw_emotions.items()
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    dominant = emotions_sorted[0]

    return {
        "dominant_emotion": dominant["emotion"],
        "confidence": dominant["confidence"],
        "mental_health_note": RELEVANCE_MAP.get(dominant["emotion"], ""),
        "all_emotions": emotions_sorted,
    }