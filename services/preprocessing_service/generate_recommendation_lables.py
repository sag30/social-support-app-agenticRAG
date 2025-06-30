import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Text, Boolean

# ─── Configuration ───────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
engine = create_engine(DB_URL, echo=False)

def run():
    # 1) Load features
    df = pd.read_sql(
        "SELECT applicant_key, income, credit_score, family_size, experience_years "
        "FROM application_features",
        engine
    )
    if df.empty:
        print("⚠ application_features is empty. Run feature_engineering.py first.")
        return

    records = []
    for _, row in df.iterrows():
        key    = row["applicant_key"]
        income = row["income"] or 0
        cs     = row["credit_score"] or 0
        fam    = row["family_size"] or 0
        exp    = row["experience_years"] or 0

        # Business rules (example thresholds)
        upskill    = (income < 25000) and (cs > 600)
        stipend    = (fam >= 4)
        counseling = (exp >= 5)

        records.append({
            "applicant_key": key,
            "upskilling_grant": upskill,
            "stipend": stipend,
            "counseling_voucher": counseling
        })

    df_labels = pd.DataFrame(records)

    # 2) Write to recommendation_labels using SQLAlchemy types
    df_labels.to_sql(
        "recommendation_labels",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "applicant_key": Text(),
            "upskilling_grant": Boolean(),
            "stipend": Boolean(),
            "counseling_voucher": Boolean()
        }
    )
    print(f"✅ Generated labels for {len(df_labels)} applicants.")

if __name__ == "__main__":
    run()
