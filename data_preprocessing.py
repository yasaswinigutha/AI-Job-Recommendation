import pandas as pd

df = pd.read_csv("dataset/jobs.csv")

print("First five rows:")
print(df.head())

print("\nNumber of job roles:", df["job_role"].nunique())

print("\nMissing values:")
print(df.isnull().sum())