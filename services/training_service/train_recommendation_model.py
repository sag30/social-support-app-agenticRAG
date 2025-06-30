import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, hamming_loss

# ─── Load preprocessed data ─────────────────────────────────────────────────────
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_test  = np.load("data/processed/y_test.npy")

# ─── Train multi‐output classifier ───────────────────────────────────────────────
base  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(base)
model.fit(X_train, y_train)

# ─── Evaluate ────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(classification_report(
    y_test, y_pred,
    target_names=["upskilling_grant","stipend","counseling_voucher"]
))

# load X_proc and y (instead of X_train/X_test—no split)
X_proc = np.load("data/processed/X_proc.npy") 
y      = np.load("data/processed/y.npy")

scoring = {
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "hamming": make_scorer(hamming_loss)
}

cv_results = cross_validate(
    MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)),
    X_proc,
    y,
    cv=3,
    scoring=scoring
)

print("CV F1-macro:", cv_results["test_f1_macro"])
print("CV Hamming:", cv_results["test_hamming"])

# ─── Persist model ────────────────────────────────────────────────────────────────
joblib.dump(model, "models/recs_model.pkl")
print("✅ Model trained and saved to models/recs_model.pkl")
