import pathlib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

current_dir = pathlib.Path(__file__).parent
df = pd.read_csv(current_dir / "data" / "SMSSpamCollection", sep="\t", names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

for v in [TfidfVectorizer, CountVectorizer]:
    vectorizer = v(max_features=1000)
    messages = vectorizer.fit_transform(df["message"])
    print("vectorizer:", v)
    print("first row , all columns:")
    print(messages[0, :])
    print("feature names out, at 888th position:")
    print(vectorizer.get_feature_names_out()[888])
