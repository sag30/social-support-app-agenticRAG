# Social Support AI Case Study

This repository contains a prototype for an AI-driven social support application. It demonstrates:

- **Multimodal Ingestion**: Bank statements, Emirates ID, resumes, assets/liabilities spreadsheets, and credit reports.
- **Data Storage**: Postgres for structured features, ChromaDB for embeddings.
- **Feature Engineering**: Numeric and demographic features plus text embeddings.
- **Model Training**: Multi-output RandomForest classifier for enablement recommendations.
- **Retrieval-Augmented Chat**: Streamlit UI with RAG over user-uploaded documents.
- **Local LLM Hosting**: Ollama for generative Q&A.

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Ingestion
      Raw["Raw Uploads"] --> ETL[ETL Pipeline]
      ETL --> Processed["Processed Files"]
    end
    Processed --> DBIngest[DB Ingestion]
    DBIngest --> Postgres[(Postgres)]
    DBIngest --> Chroma[(ChromaDB)]
    Postgres --> FeatEng[Feature Engineering]
    FeatEng --> AppFeat[application_features]
    Chroma --> RAG[RAG Q&A]
    AppFeat --> Train[Model Training]
    Train --> Model[(models/recs_model.pkl)]
    UI["Streamlit App"] --> RAG
    UI --> Model
    API["FastAPI"] --> Model
    UI & API --> User
```
## Prerequisites

- **OS**: MacOS / Linux / Windows with Python 3.10

- **PostgreSQL**: 12+

- **Ollama**: Installed and models (e.g., mistral) downloaded

- **Environment Variables**:

-- OPENAI_API_KEY: for embeddings

-- DATABASE_URL: e.g. postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport

## Setup

1. Clone the repo

git clone https://github.com/sag30/social-support-ai.git
cd social-support-ai

2. Create & activate Python venv

python3.10 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

4. Initialize the database

psql -U postgres -c "CREATE USER socialuser WITH PASSWORD 'socialpass';"
psql -U postgres -c "CREATE DATABASE socialsupport OWNER socialuser;"
psql -U socialuser -d socialsupport -f schema.sql

5. Configure environment

export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport

## Data Pipeline

1. Place raw uploads in data/raw/ (bank_statements, EmiratesID images, resumes, Excel sheets, credit reports).

2. Run ETL pipeline to OCR/convert files:
python etl_pipeline.py
This produces data/processed/manifest.json and outputs under data/processed/.

3. Ingest into Postgres & ChromaDB:
python db_ingest.py
python chroma_ingest.py

4. Feature engineering:
python feature_engineering.py

5. Prepare training data (impute, scale, embeddings concat):
python prepare_training_data.py

6. Train recommendation model:
python train_recommendation_model.py
Models and preprocessor are saved in models/.

## Running the App

1. **Streamlit UI**

streamlit run app.py

Eligibility Checker: fill basic numeric fields.

Document Upload & RAG Chat: upload files, then ask questions about your documents.

2. **FastAPI Recommendation Endpoint**

uvicorn recommendation_api:app --reload --port 8000

Endpoint: GET /recommend/{applicant_key} returns JSON with recommended programs and scores.

-- Generated on 2025-06-30