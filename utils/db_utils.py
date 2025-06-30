import os
import pandas as pd
from sqlalchemy import create_engine, text

# ─── Database connection ───────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
engine = create_engine(DB_URL, echo=False)

# ─── Fetch training data ───────────────────────────────────────────────────────
def get_training_data():
    """
    Retrieves feature matrix X and label matrix Y by joining
    application_features and recommendation_labels on applicant_key.
    """
    query = """
    SELECT
      af.applicant_key,
      af.income,
      af.net_worth,
      af.credit_score,
      af.age,
      af.experience_years,
      af.family_size,
      rl.upskilling_grant::int   AS y_upskill,
      rl.stipend::int            AS y_stipend,
      rl.counseling_voucher::int AS y_counsel
    FROM application_features af
    JOIN recommendation_labels rl
      ON rl.applicant_key = af.applicant_key;
    """
    df = pd.read_sql(query, engine)
    # Feature columns
    feature_cols = [
        'income', 'net_worth', 'credit_score',
        'age', 'experience_years', 'family_size'
    ]
    # Label columns
    label_cols = ['y_upskill', 'y_stipend', 'y_counsel']
    X = df[feature_cols]
    Y = df[label_cols]
    return X, Y

# ─── Persist recommendations ───────────────────────────────────────────────────
def save_recommendations(applicant_id: int, recs: dict):
    """
    Inserts chosen programs and scores into recommendations table.
    recs: {"Upskilling Grant":0.82, ...}
    """
    with engine.begin() as conn:
        for program, score in recs.items():
            conn.execute(
                text("""
                INSERT INTO recommendations(applicant_id, program, score, created_at)
                VALUES (:app_id, :prog, :score, NOW())
                """),
                {"app_id": applicant_id, "prog": program, "score": score}
            )