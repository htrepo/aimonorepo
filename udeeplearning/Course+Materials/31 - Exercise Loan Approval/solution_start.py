import pandas as pd

df = pd.read_csv("data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]
