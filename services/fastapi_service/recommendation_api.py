from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
engine = create_engine(DB_URL, echo=False)

app = FastAPI()

preprocessor = joblib.load("models/preprocessor.pkl")
model        = joblib.load("models/recs_model.pkl")

class RecResponse(BaseModel):
    applicant_key: str
    eligible: bool
    recommendations: dict

def load_features_from_db(applicant_key: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT income, net_worth, credit_score, age, experience_years, family_size
                  FROM application_features
                 WHERE applicant_key = :key
            """),
            {"key": applicant_key}
        ).fetchone()
    if row is None:
        raise Exception(f"Applicant '{applicant_key}' not found")
    cols = ["income","net_worth","credit_score","age","experience_years","family_size"]
    return dict(zip(cols, row))

def get_recommendations(applicant_key: str):
    try:
        feat = load_features_from_db(applicant_key)
        print("Shruti: In get_recommendations(), Loaded features:", feat)
    except Exception as e:
        return {
            "eligible": False,
            "recommendations": {},
            "error": str(e)
        }
    df = pd.DataFrame([feat])
    X_proc = preprocessor.transform(df)
    probs  = model.predict_proba(X_proc)
    labels = ["Upskilling Grant","Stipend","Career Counseling"]
    scores = {}
    for i, label_name in enumerate(labels):
        class_probs = probs[i][0]
        classes     = model.estimators_[i].classes_
        if 1 in classes:
            idx = list(classes).index(1)
            scores[label_name] = float(class_probs[idx])
        else:
            scores[label_name] = 0.0
    eligible = any(score > 0.5 for score in scores.values())
    return {
        "eligible": eligible,
        "recommendations": scores
    }

@app.get("/recommend/{applicant_key}", response_model=RecResponse)
def recommend(applicant_key: str):
    recs = get_recommendations(applicant_key)
    return RecResponse(
        applicant_key=applicant_key,
        eligible=recs["eligible"],
        recommendations=recs["recommendations"]
    )

# ─── Run the API server ────────────────────────────────────────────────────────
# To run the server, use the command:
# uvicorn recommendation_api:app --reload --port 8000