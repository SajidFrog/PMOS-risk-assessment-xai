import json
from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ml_engine.predictor import get_explainer

# In-memory longitudinal screening log for session tracking
SCREENING_HISTORY = []


def index(request):
    return render(request, "index.html")


@csrf_exempt
def assess_risk(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        patient_label = str(payload.get("patient_label", "Self-Screening Check-In")).strip() or "Self-Screening Check-In"

        features = {
            "bmi": float(payload.get("bmi", 24.5)),
            "fasting_glucose": float(payload.get("fasting_glucose", 92.0)),
            "homa_ir": float(payload.get("homa_ir", 1.8)),
            "lh_fsh_ratio": float(payload.get("lh_fsh_ratio", 1.4)),
            "free_testosterone": float(payload.get("free_testosterone", 2.1)),
            "cycle_irregularity_days": float(payload.get("cycle_irregularity_days", 4.0)),
            "triglycerides": float(payload.get("triglycerides", 115.0)),
            "hirsutism_score": float(payload.get("hirsutism_score", 5.0)),
        }

        explainer = get_explainer()
        result = explainer.predict_and_explain(features)

        record = {
            "id": len(SCREENING_HISTORY) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "patient_label": patient_label,
            "inputs": features,
            **result,
        }
        SCREENING_HISTORY.insert(0, record)
        return JsonResponse(record, status=200)

    except (ValueError, TypeError) as exc:
        return JsonResponse({"error": f"Invalid biomarker input: {exc}"}, status=400)
    except Exception as exc:
        return JsonResponse({"error": f"Inference failed: {exc}"}, status=500)


def symptom_history(request):
    return JsonResponse({"records": SCREENING_HISTORY}, status=200)