"""
FastAPI service for the credit risk cascade: Default Probability -> Loan
Grade -> Interest Rate. Run locally with:

    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive test UI.
"""
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(
    title="India Credit Risk Cascade API",
    description="Model 1 (default probability) -> Model 2 (loan grade) -> Model 3 (interest rate)",
    version="0.1.0",
)

# Allow browser-based frontends (your dummy test page, and later your real
# React dashboard) to call this API from a different origin. Wide open here
# since this is a student project behind no auth yet; tighten allow_origins
# to your actual frontend's domain once you have one, e.g.
# ["https://your-frontend.vercel.app"] instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
model1 = joblib.load(ARTIFACT_DIR / "model1_default.joblib")
model2 = joblib.load(ARTIFACT_DIR / "model2_grade.joblib")
model3 = joblib.load(ARTIFACT_DIR / "model3_rate.joblib")
GRADE_CLASSES = joblib.load(ARTIFACT_DIR / "grade_classes.joblib")

CATEGORICAL_FEATURES = [
    "state", "city_tier", "rural_urban", "gender",
    "employment_type", "sector", "loan_purpose",
]
NUMERIC_FEATURES = [
    "age", "years_at_current_job", "monthly_income", "num_dependents",
    "loan_amount_requested", "loan_tenure_months", "existing_emi",
    "foir_existing", "loan_to_income_ratio", "disposable_income",
    "cibil_score", "credit_history_months", "num_active_loans",
    "credit_utilization", "num_enquiries_6m", "dpd_max_12m",
    "loan_stacking_count_90d", "bounce_count_12m", "avg_bank_balance",
    "upi_txn_freq_monthly", "upi_txn_volatility", "gst_compliance_score",
    "sector_risk_score",
]

# ---------------------------------------------------------------------------
# SHAP explainability (Model 1 -- "why this default risk assessment")
# ---------------------------------------------------------------------------
# TreeExplainer reads the tree structure directly -- fast and exact for
# HistGradientBoostingClassifier, no extra background dataset needed.
_shap_explainer = shap.TreeExplainer(model1.named_steps["clf"])


def _map_to_original_feature(transformed_name: str) -> str:
    """Map a post-preprocessing column name (e.g. 'cat__state_Maharashtra',
    'num__missingindicator_cibil_score') back to the original applicant
    field name, so SHAP output is readable without knowing the pipeline
    internals."""
    if transformed_name.startswith("num__missingindicator_"):
        return transformed_name.replace("num__missingindicator_", "") + " (missing?)"
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "")
    if transformed_name.startswith("cat__"):
        rest = transformed_name.replace("cat__", "")
        for col in CATEGORICAL_FEATURES:
            if rest.startswith(col + "_"):
                return col
        return rest
    return transformed_name


def explain_default_risk(applicant_df: pd.DataFrame, top_n: int = 5):
    """Returns the top_n features driving Model 1's default-probability
    prediction for this one applicant, as signed contributions.
    Positive impact = pushed default risk UP. Negative = pushed it DOWN."""
    pre = model1.named_steps["pre"]
    X_transformed = pre.transform(applicant_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    feature_names = list(pre.get_feature_names_out())

    shap_values = _shap_explainer.shap_values(X_transformed)
    # Different shap/sklearn version combos return either a list of arrays
    # (one per class) or a single array for the positive class -- handle both.
    if isinstance(shap_values, list):
        row_values = shap_values[1][0]
    else:
        row_values = np.asarray(shap_values)[0]

    grouped = {}
    for name, val in zip(feature_names, row_values):
        original = _map_to_original_feature(name)
        grouped[original] = grouped.get(original, 0.0) + float(val)

    top_factors = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return [{"feature": f, "impact": round(v, 4)} for f, v in top_factors]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ApplicantRequest(BaseModel):
    state: str = Field(..., example="Maharashtra")
    city_tier: str = Field(..., example="Tier-1")
    rural_urban: str = Field(..., example="urban")
    gender: str = Field(..., example="female")
    employment_type: str = Field(..., example="salaried")
    sector: str = Field(..., example="BFSI")
    loan_purpose: str = Field(..., example="personal")

    age: int = Field(..., example=32)
    years_at_current_job: float = Field(..., example=4.5)
    monthly_income: float = Field(..., example=45000)
    num_dependents: int = Field(..., example=2)
    loan_amount_requested: float = Field(..., example=300000)
    loan_tenure_months: int = Field(..., example=36)
    existing_emi: float = Field(0.0, example=0.0)
    foir_existing: float = Field(0.0, example=0.0)
    loan_to_income_ratio: Optional[float] = Field(None, example=0.55)
    disposable_income: Optional[float] = Field(None, example=45000)

    cibil_score: Optional[float] = Field(None, example=720)
    credit_history_months: int = Field(0, example=48)
    num_active_loans: int = Field(0, example=1)
    credit_utilization: float = Field(0.0, example=0.2)
    num_enquiries_6m: int = Field(0, example=1)
    dpd_max_12m: int = Field(0, example=0)
    loan_stacking_count_90d: int = Field(0, example=0)
    bounce_count_12m: int = Field(0, example=0)
    avg_bank_balance: float = Field(..., example=30000)
    upi_txn_freq_monthly: int = Field(..., example=55)
    upi_txn_volatility: float = Field(..., example=0.3)
    gst_compliance_score: Optional[float] = Field(None, example=None)
    sector_risk_score: float = Field(..., example=0.45)

    def to_dataframe(self) -> pd.DataFrame:
        data = self.dict()
        # Derive fields the caller didn't supply, using the same formulas
        # as the dataset's data dictionary, so the API is usable with a
        # minimal payload.
        if data["loan_to_income_ratio"] is None:
            data["loan_to_income_ratio"] = data["loan_amount_requested"] / (
                data["monthly_income"] * 12
            )
        if data["disposable_income"] is None:
            data["disposable_income"] = data["monthly_income"] - data["existing_emi"]
        return pd.DataFrame([data])


class PredictionResponse(BaseModel):
    default_probability: float
    loan_grade: str
    grade_probabilities: dict
    interest_rate_pct: float
    credit_risk_score: int
    expected_roi_pct: float


# CIBIL-style scale: 300 + 600 x (1 - default_probability) x grade_weight.
# grade_weight is a placeholder monotonic decay (A best, E worst) -- swap in
# your report's Section 8 values here if they specify exact numbers.
GRADE_WEIGHT = {"A": 1.0, "B": 0.85, "C": 0.7, "D": 0.55, "E": 0.4}
COST_OF_CAPITAL_PCT = 7.0  # subtracted from expected ROI


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


class ExplainResponse(BaseModel):
    top_factors: list


@app.post("/explain", response_model=ExplainResponse)
def explain(applicant: ApplicantRequest):
    """Returns the top factors driving this applicant's default-risk score.
    'impact' > 0 means that feature pushed risk UP; < 0 means it pushed
    risk DOWN. Frontend just needs to render this as a bar list -- no SHAP
    knowledge required on that side."""
    try:
        X_base = applicant.to_dataframe()
        factors = explain_default_risk(X_base)
        return ExplainResponse(top_factors=factors)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantRequest):
    try:
        X_base = applicant.to_dataframe()

        # Model 1: default probability
        default_proba = float(model1.predict_proba(X_base[CATEGORICAL_FEATURES + NUMERIC_FEATURES])[0, 1])

        # Model 2: loan grade (needs Model 1's output as an extra feature)
        X_m2 = X_base.copy()
        X_m2["default_proba"] = default_proba
        grade_pred = model2.predict(X_m2[CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["default_proba"]])[0]
        grade_proba = model2.predict_proba(X_m2[CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["default_proba"]])[0]
        grade_proba_dict = {cls: float(p) for cls, p in zip(GRADE_CLASSES, grade_proba)}

        # Model 3: interest rate (needs Model 1 + Model 2 outputs as features)
        X_m3 = X_m2.copy()
        for cls, p in grade_proba_dict.items():
            X_m3[f"grade_proba_{cls}"] = p
        m3_cols = (
            CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["default_proba"]
            + [f"grade_proba_{c}" for c in GRADE_CLASSES]
        )
        rate_pred = float(model3.predict(X_m3[m3_cols])[0])

        # Derived, non-ML fields -- pure formulas on top of the 3 model
        # outputs above, per the dataset's data dictionary (Sec 8).
        credit_risk_score = round(
            300 + 600 * (1 - default_proba) * GRADE_WEIGHT[grade_pred]
        )
        expected_roi_pct = rate_pred * (1 - default_proba) - COST_OF_CAPITAL_PCT

        return PredictionResponse(
            default_probability=round(default_proba, 4),
            loan_grade=grade_pred,
            grade_probabilities={k: round(v, 4) for k, v in grade_proba_dict.items()},
            interest_rate_pct=round(rate_pred, 2),
            credit_risk_score=credit_risk_score,
            expected_roi_pct=round(expected_roi_pct, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
