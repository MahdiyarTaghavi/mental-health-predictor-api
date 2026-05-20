import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import predict

sample_input = {
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
    "seek_help": "No",
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

result = predict(sample_input)

print("\n" + "═" * 50)
print("  Prediction Result")
print("═" * 50)
print(f"  Prediction  : {result['label']}")
print(f"  Confidence  : {result['confidence']}%")
print(f"  Model used  : {result['model_used']}")
print(f"  Model AUC   : {result['model_auc']}")
print(f"\n  Top global features:")
for f in result["top_global_influential_features"]:
    print(f"    - {f}")
print(f"\n  SHAP explanation (baseline: {result['shap_explanation']['baseline']}):")
for contrib in result["shap_explanation"]["top_contributors"]:
    print(f"    {contrib['feature']:<30} {contrib['contribution']:>+.4f}")
print("═" * 50)