DROP TABLE IF EXISTS application_features, recommendations, recommendation_labels,
                     application_logs, resumes, assets_liabilities,
                     credit_reports, bank_transactions, raw_documents CASCADE;
                     
CREATE TABLE IF NOT EXISTS raw_documents (
  id SERIAL PRIMARY KEY,
  applicant_key  TEXT,
  filename TEXT,
  file_type TEXT,
  sheet_name TEXT
);

CREATE TABLE IF NOT EXISTS bank_transactions (
  doc_id INTEGER REFERENCES raw_documents(id),
  txn_date DATE,
  description TEXT,
  amount NUMERIC,
  balance_after NUMERIC
);
CREATE TABLE IF NOT EXISTS credit_reports (
  doc_id INTEGER REFERENCES raw_documents(id),
  credit_score INTEGER,
  utilization_pct NUMERIC,
  inquiries_last_12m INTEGER
);
CREATE TABLE IF NOT EXISTS assets_liabilities (
  doc_id INTEGER REFERENCES raw_documents(id),
  category TEXT,
  value NUMERIC
);
CREATE TABLE IF NOT EXISTS resumes (
  doc_id INTEGER REFERENCES raw_documents(id),
  dob DATE,
  nationality TEXT,
  total_experience_years INTEGER,
  current_position TEXT
);
CREATE TABLE IF NOT EXISTS application_logs (
  id SERIAL PRIMARY KEY,
  submitted_at TIMESTAMP,
  income NUMERIC,
  employment_years INTEGER,
  family_size INTEGER,
  assets_value NUMERIC,
  model_pred BOOLEAN
);
CREATE TABLE IF NOT EXISTS recommendation_labels (
  applicant_id INTEGER PRIMARY KEY,
  upskilling_grant BOOLEAN,
  stipend BOOLEAN,
  counseling_voucher BOOLEAN,
  applicant_key TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
  id SERIAL PRIMARY KEY,
  applicant_id INTEGER,
  program TEXT,
  score NUMERIC,
  created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS application_features (
  applicant_key TEXT PRIMARY KEY,
  income NUMERIC,
  net_worth NUMERIC,
  credit_score INTEGER,
  age INTEGER,
  experience_years INTEGER,
  family_size INTEGER
);
