import io
from PIL import Image


# ── / ────────────────────────────────────────────────────────────────────

def test_root_returns_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Mental Health Prediction is up and running",
        "version": "1.0",
        "status": "healthy"
    }

# ── /model-info ────────────────────────────────────────────────────────────────

def test_model_info_returns_200(client):
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_has_required_keys(client):
    response = client.get("/model-info")
    body = response.json()
    assert "model_name" in body
    assert "auc_score" in body
    assert "top_5_features" in body


def test_model_info_auc_is_valid(client):
    response = client.get("/model-info")
    auc = response.json()["auc_score"]
    assert 0.0 <= auc <= 1.0


# ── /predict — happy path ──────────────────────────────────────────────────────

def test_predict_returns_200(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_has_required_keys(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert "prediction" in body
    assert "label" in body
    assert "confidence" in body
    assert "shap_explanation" in body
    assert "model_used" in body
    assert "model_auc" in body


def test_predict_prediction_is_binary(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    prediction = response.json()["prediction"]
    assert prediction in [0, 1]


def test_predict_confidence_in_valid_range(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 100.0


def test_predict_label_matches_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    if body["prediction"] == 1:
        assert body["label"] == "Likely to seek treatment"
    else:
        assert body["label"] == "Unlikely to seek treatment"


def test_predict_shap_has_required_keys(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    shap = response.json()["shap_explanation"]
    assert "baseline" in shap
    assert "top_contributors" in shap


def test_predict_shap_contributors_structure(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    contributors = response.json()["shap_explanation"]["top_contributors"]
    assert len(contributors) > 0
    for item in contributors:
        assert "feature" in item
        assert "contribution" in item


# ── /predict — sad path ────────────────────────────────────────────────────────

def test_predict_rejects_invalid_gender(client, valid_payload):
    payload = {**valid_payload, "Gender": "InvalidGender"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_age_too_low(client, valid_payload):
    payload = {**valid_payload, "Age": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_age_too_high(client, valid_payload):
    payload = {**valid_payload, "Age": 200}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, valid_payload):
    payload = {**valid_payload}
    del payload["family_history"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_fields(client, valid_payload):
    payload = {**valid_payload, "unexpected_field": "hacked"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ── /analyze-emotion — sad path ────────────────────────────────────────────────

def test_analyze_emotion_rejects_wrong_content_type(client, fake_pdf):
    response = client.post(
        "/analyze-emotion",
        files={"file": ("document.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400


def test_analyze_emotion_rejects_oversized_image(client, oversized_image):
    response = client.post(
        "/analyze-emotion",
        files={"file": ("large.jpg", oversized_image, "image/jpeg")},
    )
    assert response.status_code == 413


def test_analyze_emotion_rejects_no_face(client, blank_image):
    response = client.post(
        "/analyze-emotion",
        files={"file": ("blank.jpg", blank_image, "image/jpeg")},
    )
    assert response.status_code == 400