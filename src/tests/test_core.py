from core.ml.model import predict


VALID_INPUT = {
    "Age": 28,
    "Gender": "Male",
    "self_employed": "No",
    "family_history": "Yes",
    "work_interfere": "Sometimes",
    "no_employees": "26-100",
    "remote_work": "No",
    "tech_company": "Yes",
    "benefits": "Yes",
    "care_options": "Not sure",
    "wellness_program": "No",
    "seek_help": "Yes",
    "anonymity": "Yes",
    "leave": "Somewhat easy",
    "mental_health_consequence": "No",
    "phys_health_consequence": "No",
    "coworkers": "Some of them",
    "supervisor": "Yes",
    "mental_health_interview": "No",
    "phys_health_interview": "Maybe",
    "mental_vs_physical": "Yes",
    "obs_consequence": "No",
}


def test_predict_returns_dict():
    result = predict(VALID_INPUT)
    assert isinstance(result, dict)


def test_predict_has_all_keys():
    result = predict(VALID_INPUT)
    assert "prediction" in result
    assert "label" in result
    assert "confidence" in result
    assert "shap_explanation" in result
    assert "model_used" in result
    assert "model_auc" in result


def test_predict_prediction_is_binary():
    result = predict(VALID_INPUT)
    assert result["prediction"] in [0, 1]


def test_predict_confidence_in_range():
    result = predict(VALID_INPUT)
    assert 0.0 <= result["confidence"] <= 100.0


def test_predict_label_matches_prediction():
    result = predict(VALID_INPUT)
    if result["prediction"] == 1:
        assert result["label"] == "Likely to seek treatment"
    else:
        assert result["label"] == "Unlikely to seek treatment"


def test_predict_shap_baseline_is_float():
    result = predict(VALID_INPUT)
    assert isinstance(result["shap_explanation"]["baseline"], float)


def test_predict_shap_contributors_not_empty():
    result = predict(VALID_INPUT)
    assert len(result["shap_explanation"]["top_contributors"]) > 0


def test_predict_shap_contributor_keys():
    result = predict(VALID_INPUT)
    for item in result["shap_explanation"]["top_contributors"]:
        assert "feature" in item
        assert "contribution" in item


def test_predict_model_auc_is_valid():
    result = predict(VALID_INPUT)
    assert 0.0 <= result["model_auc"] <= 1.0