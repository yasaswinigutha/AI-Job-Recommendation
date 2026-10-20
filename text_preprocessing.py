
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load the dataset
df = pd.read_csv("dataset/jobs.csv")

# Step 2: Combine skills and description
df["combined_text"] = (
    df["skills"] + " " + df["description"]
)

print("Original text:")
print(df["combined_text"].head())

# Step 3: Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Step 4: Convert text into numerical features
X = vectorizer.fit_transform(df["combined_text"])

print("\nTF-IDF matrix shape:")
print(X.shape)

# Step 5: Display some words learned by TF-IDF
print("\nFirst 20 vocabulary words:")
print(vectorizer.get_feature_names_out()[:20])

# Step 6: Display numerical values for the first job
print("\nFirst job's TF-IDF values:")
print(X.toarray()[0])
