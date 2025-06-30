import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date
from dateutil.relativedelta import relativedelta

# ─── Configuration ───────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
engine = create_engine(DB_URL, echo=False)

def compute_age(dob):
    return relativedelta(date.today(), dob).years

def run():
    # 1) Get all distinct applicants
    df_apps = pd.read_sql("SELECT DISTINCT applicant_key FROM raw_documents", engine)
    if df_apps.empty:
        print("⚠ No applicants found in raw_documents.")
        return

    records = []
    for app in df_apps["applicant_key"]:
        # 2) Fetch all doc_ids + sheet_names for this applicant
        df_docs = pd.read_sql(
            text("SELECT id, sheet_name FROM raw_documents WHERE applicant_key = :app"),
            engine,
            params={"app": app}
        )
        if df_docs.empty:
            continue
        doc_ids = df_docs["id"].tolist()

        # 3) Income estimate from bank_transactions
        tx = pd.read_sql(
            text("SELECT * FROM bank_transactions WHERE doc_id = ANY(:ids)"),
            engine,
            params={"ids": doc_ids}
        )
        if tx.empty:
            income = 0
        else:
            # sum up all credit transactions (Salary, Dividend, etc.) as monthly income proxy
            credits = tx[tx["amount"] > 0]["amount"]
            income = credits.sum() if not credits.empty else 0

        # 4) Net worth using sheet_name to separate assets vs liabilities
        # Document IDs for "Assets" and "Liabilities" sheets
        asset_ids = df_docs[df_docs["sheet_name"].str.lower() == "assets"]["id"].tolist()
        liability_ids = df_docs[df_docs["sheet_name"].str.lower() == "liabilities"]["id"].tolist()
        # Query values
        assets_df = pd.read_sql(
            text("SELECT value FROM assets_liabilities WHERE doc_id = ANY(:ids)"),
            engine,
            params={"ids": asset_ids}
        )
        liabs_df = pd.read_sql(
            text("SELECT value FROM assets_liabilities WHERE doc_id = ANY(:ids)"),
            engine,
            params={"ids": liability_ids}
        )
        assets_val = assets_df["value"].sum() if not assets_df.empty else 0
        liabs_val = liabs_df["value"].sum() if not liabs_df.empty else 0
        net_worth = assets_val - liabs_val

        # 5) Credit health from credit_reports
        cr = pd.read_sql(
            text("SELECT * FROM credit_reports WHERE doc_id = ANY(:ids)"),
            engine,
            params={"ids": doc_ids}
        )
        credit_score = int(cr["credit_score"].iloc[0]) if not cr.empty else None

        # 6) Demographics & experience from resumes
        rv = pd.read_sql(
            text("SELECT dob, total_experience_years FROM resumes WHERE doc_id = ANY(:ids)"),
            engine,
            params={"ids": doc_ids}
        )
        if rv.empty:
            age = None
            exp_years = None
        else:
            dob_raw = rv["dob"].iloc[0]
            dob = pd.to_datetime(dob_raw).date() if dob_raw is not None else None
            age = compute_age(dob) if dob else None
            exp_years = int(rv["total_experience_years"].iloc[0]) if rv["total_experience_years"].iloc[0] is not None else None

        # 7) Family size (from application_logs, placeholder)
        family_size = None

        records.append({
            "applicant_key": app,
            "income": income,
            "net_worth": net_worth,
            "credit_score": credit_score,
            "age": age,
            "experience_years": exp_years,
            "family_size": family_size
        })

    # 8) Persist to application_features
    df_feat = pd.DataFrame(records)
    if df_feat.empty:
        print("⚠ No feature records generated.")
    else:
        df_feat.to_sql(
            "application_features",
            engine,
            if_exists="replace",
            index=False
        )
        print(f"✅ Feature engineering complete: {len(df_feat)} applicants processed.")

if __name__ == "__main__":
    run()
