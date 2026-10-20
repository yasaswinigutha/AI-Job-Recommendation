
import pandas as pd
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# 1. Locate and load the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "jobs.csv"

df = pd.read_csv(DATASET_PATH)

# 2. Check the required columns
required_columns = ["skills", "job_role"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}. "
            f"Available columns: {list(df.columns)}"
        )

# 3. Clean the dataset
df = df.dropna(subset=["skills", "job_role"]).copy()

df["skills"] = df["skills"].astype(str).str.strip()
df["job_role"] = df["job_role"].astype(str).str.strip()

df = df[
    (df["skills"] != "") &
    (df["job_role"] != "")
].copy()

# 4. Separate inputs and target labels
X = df["skills"]
y = df["job_role"]

if len(df) < 2 or y.nunique() < 2:
    raise ValueError(
        "The dataset must contain at least two different job roles."
    )

print("Dataset loaded successfully!")
print("Total examples:", len(df))
print("Different job roles:", y.nunique())

# 5. Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train the model
model.fit(X, y)

print("\nModel training completed!")

# 7. Get skills from the user
user_skills = input(
    "\nInput skills (e.g., Python SQL Django REST API Git): "
).strip()

if not user_skills:
    raise ValueError("Please enter your skills.")

# 8. Recommend a job role
recommended_role = model.predict([user_skills])[0]

print("\nRecommended job role:", recommended_role)

# 9. Cross-validation
class_counts = y.value_counts()
min_class_count = int(class_counts.min())

if min_class_count >= 2:
    cv_folds = min(5, min_class_count)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_folds,
        scoring="accuracy"
    )

    print("\nCross-validation scores:", scores)
    print(
        "Average cross-validation accuracy: {:.2f}%".format(
            scores.mean() * 100
        )
    )
else:
    print("\nCross-validation skipped.")
    print(
        "At least one job role has fewer than 2 examples. "
        "Add more examples for each role to evaluate the model."
    )
