import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from chromadb import PersistentClient
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib


# ─── Config ────────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL","postgresql+psycopg2://socialuser:socialpass@localhost:5432/socialsupport")
engine = create_engine(DB_URL, echo=False)

# ─── ChromaDB client ───────────────────────────────────────────────────────────
client = PersistentClient(path="chromadb_data")
coll   = client.get_collection("support_documents")

# ─── 1) Load numeric features & labels ─────────────────────────────────────────
df_feat = pd.read_sql("SELECT * FROM application_features", engine).set_index("applicant_key")
df_lbl  = pd.read_sql("SELECT * FROM recommendation_labels", engine).set_index("applicant_key")

print("Loaded features shape:", df_feat.shape)
print("Loaded labels shape:", df_lbl.shape)
print(df_feat.index) # List of applicant keys

# ─── 2) Fetch & aggregate embeddings per applicant_key ─────────────────────────
embeddings = []
emb_dim = None

for app_key in df_feat.index: 
    # Pull back only chunks whose metadata.applicant_key == this key
    resp = coll.get(
            where={ "applicant_key": { "$eq": app_key } },
            include=["embeddings"]
            )
    embs = resp.get("embeddings", [])
    print(len(embs), "embeddings found for", app_key)
    if embs is not None and len(embs) > 0:
        arr = np.array(embs)
        print("Shape of embeddings array:", arr.shape)
        print(arr)
        embeddings.append(arr.mean(axis=0)) # average embeddings
        emb_dim = arr.shape[1] # get embedding dimension
    else:
        # pad with zeros if no embeddings found
        if emb_dim is None:
            # we need at least one real sample to know dim
            sample = coll.get(include=["embeddings"])["embeddings"]
            if sample and len(sample[0]) > 0:
                emb_dim = len(sample[0])
            else:
                emb_dim = 1536  # fallback
        embeddings.append(np.zeros(emb_dim)) # pad with zeros

emb_df = pd.DataFrame(embeddings, index=df_feat.index).add_prefix("emb_")
print("Embeddings DataFrame shape:", emb_df.shape)

# ─── 3) Merge application_features data + embeddings; get labels ────────────────────────────────
dfX = pd.concat([df_feat, emb_df], axis=1, join="inner") 
dfY = df_lbl.loc[dfX.index, ['upskilling_grant','stipend','counseling_voucher']]  # get labels from db only for those applicants for which we have features in application_features tables.
print("Merged features shape:", dfX.shape)
print("Labels shape:", dfY.shape)

# ─── 4) Preprocessing pipeline with imputation ─────────────────────────────────
# We'll treat all six features as numeric and impute missing values.
numeric_cols = [
    "income",
    "net_worth",
    "credit_score",
    "age",
    "experience_years",
    "family_size"
]
# Pipeline for numeric features: impute then scale
numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),   # fill NaNs with median
    StandardScaler()                    # then standardize
)
preprocessor = ColumnTransformer(
    [("num", numeric_transformer, numeric_cols)],
    remainder="drop"                    # drop any other columns
)
pipeline = make_pipeline(preprocessor)

# ─── 5) Train‐test split & save ────────────────────────────────────────────────
X_proc = pipeline.fit_transform(dfX)
y      = dfY.values

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape, y_train.shape)
print("Test data shape:", X_test.shape, y_test.shape)
print(X_train[:5])  # Show first 5 rows of training data
print(y_train[:5])  # Show first 5 labels
print(X_test[:5])   # Show first 5 rows of test data
print(y_test[:5])    # Show first 5 test labels

# ─── 6) Save processed data and pipeline ───────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/preprocessor.pkl")
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
np.save("data/processed/X_proc.npy", X_proc)
np.save("data/processed/y.npy", y)

print("✅ Prepared final training data with embeddings and saved to disk.")
