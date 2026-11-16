import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


FEATURE_META = [
    {
        "key": "bmi",
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "normal_range": "18.5 - 24.9",
        "advice": "Sustained 5-7% weight reduction through low-glycemic nutrition significantly improves insulin sensitivity.",
    },
    {
        "key": "fasting_glucose",
        "label": "Fasting Blood Glucose",
        "unit": "mg/dL",
        "normal_range": "70 - 99",
        "advice": "Pair complex carbohydrates with lean protein and fiber to blunt post-meal glucose spikes.",
    },
    {
        "key": "homa_ir",
        "label": "HOMA-IR (Insulin Resistance)",
        "unit": "index",
        "normal_range": "0.5 - 1.9",
        "advice": "Regular resistance training and 150 minutes/week of aerobic activity directly lower insulin resistance.",
    },
    {
        "key": "lh_fsh_ratio",
        "label": "LH / FSH Hormonal Ratio",
        "unit": "ratio",
        "normal_range": "1.0 - 2.0",
        "advice": "Elevated LH/FSH ratios reflect neuroendocrine signaling imbalance; share this trend with an endocrinologist.",
    },
    {
        "key": "free_testosterone",
        "label": "Serum Free Testosterone",
        "unit": "ng/dL",
        "normal_range": "0.5 - 2.4",
        "advice": "Elevated androgen levels correlate with metabolic and ovulatory disruption; clinical hormone panel review advised.",
    },
    {
        "key": "cycle_irregularity_days",
        "label": "Menstrual Cycle Variance",
        "unit": "days",
        "normal_range": "0 - 7",
        "advice": "Track cycle lengths consistently across 3+ months to detect oligomenorrhea patterns early.",
    },
    {
        "key": "triglycerides",
        "label": "Fasting Triglycerides",
        "unit": "mg/dL",
        "normal_range": "< 150",
        "advice": "Increase omega-3 fatty acid intake and limit refined sugars to improve cardiometabolic lipid markers.",
    },
    {
        "key": "hirsutism_score",
        "label": "Clinical Hirsutism (mFG Score)",
        "unit": "pts",
        "normal_range": "0 - 7",
        "advice": "Higher modified Ferriman-Gallwey scores indicate peripheral androgen sensitivity.",
    },
]

RISK_TIERS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}


class PMOSExplainer:
    """Multi-level PMOS risk stratification model paired with SHapley Additive exPlanations."""

    def __init__(self):
        self.feature_keys = [f["key"] for f in FEATURE_META]
        self.model = self._train_calibrated_model()
        self.explainer = shap.TreeExplainer(self.model) if SHAP_AVAILABLE else None

    def _train_calibrated_model(self):
        rng = np.random.default_rng(42)
        n_samples = 750

        bmi = rng.normal(26.5, 5.2, n_samples).clip(16.0, 44.0)
        fasting_glucose = rng.normal(98.0, 18.0, n_samples).clip(65.0, 180.0)
        homa_ir = rng.normal(2.4, 1.3, n_samples).clip(0.5, 8.5)
        lh_fsh_ratio = rng.normal(1.9, 0.85, n_samples).clip(0.4, 4.8)
        free_testosterone = rng.normal(2.5, 1.2, n_samples).clip(0.3, 6.5)
        cycle_var = rng.normal(11.0, 8.5, n_samples).clip(0.0, 45.0)
        triglycerides = rng.normal(135.0, 45.0, n_samples).clip(50.0, 320.0)
        hirsutism = rng.normal(7.5, 5.0, n_samples).clip(0.0, 26.0)

        X = np.column_stack([
            bmi,
            fasting_glucose,
            homa_ir,
            lh_fsh_ratio,
            free_testosterone,
            cycle_var,
            triglycerides,
            hirsutism,
        ])

        # Clinical composite severity score across metabolic, endocrine, and ovulatory axes
        severity = (
            0.22 * ((bmi - 22.0) / 5.0)
            + 0.20 * ((homa_ir - 1.5) / 1.2)
            + 0.16 * ((fasting_glucose - 90.0) / 15.0)
            + 0.16 * ((lh_fsh_ratio - 1.5) / 0.8)
            + 0.14 * ((free_testosterone - 2.0) / 1.0)
            + 0.12 * ((cycle_var - 5.0) / 7.0)
            + 0.08 * ((triglycerides - 120.0) / 40.0)
            + 0.08 * ((hirsutism - 6.0) / 4.5)
        )

        y = np.where(severity < 0.35, 0, np.where(severity < 1.15, 1, 2))

        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            random_state=42,
        )
        clf.fit(X, y)
        return clf

    def predict_and_explain(self, feature_dict: dict) -> dict:
        row = np.array([[feature_dict[k] for k in self.feature_keys]], dtype=float)
        probs = self.model.predict_proba(row)[0]
        pred_class = int(np.argmax(probs))

        if self.explainer is not None:
            raw_shap = self.explainer.shap_values(row)
            if isinstance(raw_shap, list):
                class_shap = np.asarray(raw_shap[pred_class])[0]
            else:
                arr = np.asarray(raw_shap)
                class_shap = arr[0, :, pred_class] if arr.ndim == 3 else arr[0]
        else:
            # Fallback baseline perturbation if SHAP binary is unavailable
            baseline = np.array([[22.0, 88.0, 1.4, 1.3, 1.6, 3.0, 105.0, 4.0]])
            base_prob = self.model.predict_proba(baseline)[0][pred_class]
            class_shap = np.zeros(len(self.feature_keys))
            for idx in range(len(self.feature_keys)):
                perturbed = baseline.copy()
                perturbed[0, idx] = row[0, idx]
                class_shap[idx] = self.model.predict_proba(perturbed)[0][pred_class] - base_prob

        explanations = []
        for idx, meta in enumerate(FEATURE_META):
            val = float(row[0, idx])
            s_val = float(class_shap[idx])
            explanations.append({
                "key": meta["key"],
                "label": meta["label"],
                "unit": meta["unit"],
                "normal_range": meta["normal_range"],
                "patient_value": round(val, 2),
                "shap_value": round(s_val, 4),
                "direction": "increases_risk" if s_val > 0 else "reduces_risk",
                "advice": meta["advice"],
            })

        # Sort by strongest absolute SHAP impact
        explanations.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        top_drivers = [e["advice"] for e in explanations if e["shap_value"] > 0][:3]
        if not top_drivers:
            top_drivers = [
                "All evaluated endocrine and metabolic markers fall within low-risk bounds. Maintain current nutrition and sleep routines."
            ]

        return {
            "risk_tier": pred_class,
            "risk_label": RISK_TIERS[pred_class],
            "confidence_pct": round(float(probs[pred_class]) * 100, 1),
            "probabilities": {
                "low": round(float(probs[0]) * 100, 1),
                "moderate": round(float(probs[1]) * 100, 1),
                "high": round(float(probs[2]) * 100, 1),
            },
            "shap_explanations": explanations,
            "recommendations": top_drivers,
            "explainer_engine": "SHAP TreeExplainer" if SHAP_AVAILABLE else "Baseline Perturbation",
        }


_EXPLAINER_INSTANCE = None


def get_explainer() -> PMOSExplainer:
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = PMOSExplainer()
    return _EXPLAINER_INSTANCE