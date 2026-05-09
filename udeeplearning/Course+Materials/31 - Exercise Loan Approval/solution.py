import pandas as pd
import torch

df = pd.read_csv("data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]
df = pd.get_dummies(df, columns=["loan_intent"])
print(df.columns)

y = torch.tensor(df["loan_status"], dtype=torch.float32).reshape((-1, 1))

print(df.drop("loan_status", axis=1))
X_data = df.drop("loan_status", axis=1).astype("float32").values
print(X_data.dtype)
X = torch.tensor(X_data, dtype=torch.float32)
print(X)
