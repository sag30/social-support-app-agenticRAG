import os
import json
import pandas as pd
import re
from sqlalchemy import create_engine, text
from datetime import datetime

# ─── Configure DB connection ─────────Select * FROM bank_transactionsSelect * FROM credit_reportsSelect * FROM credit_reportsSelect * FROM raw_documents──────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
engine = create_engine(DB_URL, echo=False)


def ingest():
    # read manifest of processed files
    with open("data/processed/manifest.json", "r") as f:
        manifest = json.load(f)

    with engine.begin() as conn:
        for entry in manifest:
            print(f"INGEST: source={entry['source']}, type={entry['type']}, output={entry['output']}")
            fn = entry["source"]
            ftype = entry["type"]
            sheet = entry.get("sheet")
            out = entry["output"]

            # 1) derive applicant_key from filename (last underscore segment)
            base = os.path.splitext(fn)[0]                    # e.g. 'bank_statement_zeeshan'
            applicant_key = base.split('_')[-1].lower()        # 'zeeshan'+            # 2) record metadata
            res = conn.execute(
                text(
                    """
                    INSERT INTO raw_documents
                      (filename, file_type, sheet_name, applicant_key)
                    VALUES
                      (:fn, :ft, :sn, :ak)
                    RETURNING id
                    """
                ),
                {"fn": fn, "ft": ftype, "sn": sheet, "ak": applicant_key},
            )
            doc_id = res.scalar()

            # 2) dispatch table loads
            if ftype == "table":
                df = pd.read_csv(out)

                if "bank_statement" in fn.lower():
                    target = "bank_transactions"
                elif "credit_report" in fn.lower():
                    target = "credit_reports"
                elif "assets_liabilities" in fn.lower():
                    target = "assets_liabilities"
                else:
                    continue

                # normalize column keys for assets/liabilities
                if target == "assets_liabilities":
                    cols = {c.lower(): c for c in df.columns}
                    cat_col = next((orig for low, orig in cols.items() if "category" in low), None)
                    val_col = next((orig for low, orig in cols.items() if "value" in low), None)

                for _, row in df.iterrows():
                    if target == "bank_transactions":
                        # map raw lower-case names to actual column names
                        cols = {c.lower(): c for c in df.columns}
                        date_col   = next((orig for low, orig in cols.items() if low == "date"), None)
                        desc_col   = next((orig for low, orig in cols.items() if "desc" in low), None)
                        debit_col  = next((orig for low, orig in cols.items() if "debit" in low), None)
                        credit_col = next((orig for low, orig in cols.items() if "credit" in low), None)
                        bal_col    = next((orig for low, orig in cols.items() if "balance" in low), None)

                        for _, row in df.iterrows():
                            # parse date & description
                            txn_date = row[date_col]
                            desc     = row[desc_col]

                            # robustly parse numeric debit/credit
                            raw_credit = row[credit_col]
                            raw_debit  = row[debit_col]
                            try:
                                credit_val = float(str(raw_credit).replace(",", ""))
                            except:
                                credit_val = None
                            try:
                                debit_val = float(str(raw_debit).replace(",", ""))
                            except:
                                debit_val = None

                            # decide net amount
                            if credit_val and credit_val > 0:
                                amount = credit_val
                            elif debit_val and debit_val > 0:
                                amount = -debit_val
                            else:
                                amount = None

                            # parse balance if present
                            try:
                                balance_after = float(str(row[bal_col]).replace(",", "")) if bal_col else None
                            except:
                                balance_after = None

                            # insert into the DB
                            conn.execute(
                                text("""
                                INSERT INTO bank_transactions
                                    (doc_id, txn_date, description, amount, balance_after)
                                VALUES (:doc_id, :d, :desc, :amt, :bal)
                                """),
                                {
                                    "doc_id": doc_id,
                                    "d": txn_date,
                                    "desc": desc.strip(),
                                    "amt": amount,
                                    "bal": balance_after,
                                },
                            )

                    elif target == "credit_reports":
                        conn.execute(
                            text(
                                """
                                INSERT INTO credit_reports
                                (doc_id, credit_score, utilization_pct, inquiries_last_12m)
                                VALUES (:doc_id, :score, :util, :inq)
                                """
                            ),
                            {
                                "doc_id": doc_id,
                                "score": row.get("credit_score"),
                                "util": row.get("utilization_pct"),
                                "inq": row.get("inquiries_last_12m"),
                            },
                        )

                    elif target == "assets_liabilities":
                        conn.execute(
                            text(
                                """
                                INSERT INTO assets_liabilities
                                (doc_id, category, value)
                                VALUES (:doc_id, :cat, :val)
                                """
                            ),
                            {
                                "doc_id": doc_id,
                                "cat": row[cat_col],
                                "val": row[val_col],
                            },
                        )

                         # … after your table‐loads block …

            elif ftype == "text" and "bank_statement" in fn.lower():
                lines = open(out, encoding="utf-8").read().splitlines()
                # match: date at start, then description, then debit, credit, optional balance
                pattern = re.compile(
                    r'^(\d{2}/\d{2}/\d{4})\s+'          # 1) date dd/mm/yyyy
                    r'(.*?)\s+'                         # 2) non-greedy description
                    r'(-?[\d,]+\.\d{2})\s+'             # 3) debit or credit value
                    r'(-?[\d,]+\.\d{2})'                # 4) the other value
                    r'(?:\s+(-?[\d,]+\.\d{2}))?$'       # 5) optional balance
                )
                for line in lines:
                    m = pattern.match(line)
                    if not m:
                        continue
                    date_str, desc, val1_s, val2_s, bal_s = m.groups()
                    txn_date = datetime.strptime(date_str, "%d/%m/%Y").date()
                    # parse numbers
                    v1 = float(val1_s.replace(",", "")) if val1_s else 0.0
                    v2 = float(val2_s.replace(",", "")) if val2_s else 0.0
                    # decide which is credit vs. debit by positive sign
                    # if both positive, assume one is credit, one is balance—so treat the larger as balance
                    if v1 > 0 and v2 > 0:
                        amount = v1  if v1 <= v2 else v2
                    else:
                        # one is negative debit, one is credit
                        amount = v1 if v1 > 0 else (v2 if v2 > 0 else 0.0)
                    # balance if present
                    balance_after = float(bal_s.replace(",", "")) if bal_s else None

                    conn.execute(
                        text("""
                            INSERT INTO bank_transactions
                            (doc_id, txn_date, description, amount, balance_after)
                            VALUES
                            (:doc_id, :d, :desc, :amt, :bal)
                        """),
                        {
                            "doc_id": doc_id,
                            "d": txn_date,
                            "desc": desc.strip(),
                            "amt": amount,
                            "bal": balance_after
                        }
                    )

            # 3b) Parse credit reports from text
            elif ftype == "text" and "credit_report" in fn.lower():
                text_blob = open(out, encoding="utf-8").read()
                score_m = re.search(r'Credit\s*Score[:]?[\s]*([\d]{3})', text_blob)
                util_m  = re.search(r'Utilization\s*[:]?[\s]*([\d]{1,3})\s*%', text_blob)
                inq_m   = re.search(r'Inquiries\s*(?:last\s*12\s*months)?\s*[:]?[\s]*(\d+)', text_blob, re.IGNORECASE)

                conn.execute(
                    text("""
                      INSERT INTO credit_reports
                        (doc_id, credit_score, utilization_pct, inquiries_last_12m)
                      VALUES (:doc_id, :score, :util, :inq)
                    """),
                    {
                      "doc_id": doc_id,
                      "score": int(score_m.group(1)) if score_m else None,
                      "util":  float(util_m.group(1)) if util_m  else None,
                      "inq":   int(inq_m.group(1))  if inq_m   else None
                    },
                )

            # 3) parse and insert resumes from text docs
            elif ftype == "text" and "resume" in fn.lower():
                # read the processed resume text
                text_blob = open(out, encoding="utf-8").read()
                # extract fields via regex
                dob_match = re.search(r"Date of Birth[:]?\s*(\d{1,2} [A-Za-z]+ \d{4})", text_blob)
                nat_match = re.search(r"Nationality[:]?\s*([A-Za-z ]+)", text_blob)
                exp_match = re.search(r"(\d+)\s+years", text_blob, re.IGNORECASE)

                dob = None
                if dob_match:
                    dob = datetime.strptime(dob_match.group(1), "%d %B %Y").date()
                nationality = nat_match.group(1).strip() if nat_match else None
                exp_years = int(exp_match.group(1)) if exp_match else None

                # insert into resumes table (current_position left NULL)
                conn.execute(
                    text(
                        """
                        INSERT INTO resumes(doc_id, dob, nationality, total_experience_years, current_position)
                        VALUES (:doc_id, :dob, :nat, :exp, NULL)
                        """
                    ),
                    {"doc_id": doc_id, "dob": dob, "nat": nationality, "exp": exp_years},
                )

    print("✅ db_ingest complete")

if __name__ == "__main__":
    ingest()
