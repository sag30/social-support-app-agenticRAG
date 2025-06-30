import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Config ───────────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
engine = create_engine(DB_URL, echo=False)

# ─── Embedding function setup ─────────────────────────────────────────────────
# embed_fn = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=OPENAI_API_KEY,
#     model_name="text-embedding-ada-002"
# )

emb = OllamaEmbeddings(model="mistral")
vectordb = Chroma(
    collection_name="support_documents",
    persist_directory="chromadb_data",
    embedding_function=emb,
)

def ingest():
    manifest_path = "data/processed/manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.load(open(manifest_path, "r"))
    print(manifest)
    with engine.begin() as conn:
        for entry in manifest:
            fn = entry["source"]
            out = entry["output"]
            sheet = entry.get("sheet")

            # Lookup the document ID
            result = conn.execute(
                text(
                    """
                    SELECT id, applicant_key FROM raw_documents
                     WHERE filename = :fn
                       AND sheet_name IS NOT DISTINCT FROM :sn
                    """
                ),
                {"fn": fn, "sn": sheet}
            ).first()
            if not result:
                continue
            doc_id, app_key = result

            print(doc_id,app_key)

            # Load content for embedding
            if entry["type"] == "text":
                text_blob = open(out, encoding="utf-8").read()
            else:
                df = pd.read_csv(out)
                text_blob = df.to_string(index=False)

            # 3) Chunk and add to ChromaDB
            chunk_size = 1000
            for idx in range(0, len(text_blob), chunk_size):
                chunk_id = idx // chunk_size
                chunk    = text_blob[idx : idx + chunk_size]
                metadata = {
                    "doc_id":        doc_id,
                    "applicant_key": app_key,
                    "source":        fn,
                    "chunk_id":      chunk_id
                }
                vectordb.add_texts(texts=[chunk], metadatas=[metadata])
        vectordb.persist()

    print("✅ ChromaDB ingestion complete")

if __name__ == "__main__":
    ingest()