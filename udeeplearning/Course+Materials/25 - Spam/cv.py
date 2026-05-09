import pathlib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

current_dir = pathlib.Path(__file__).parent
df = pd.read_csv(current_dir / "data" / "SMSSpamCollection", sep="\t", names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

print("""first row , all columns:""")
print(messages[0, :])
print("""feature names out, at 888th position:""")
print(cv.get_feature_names_out()[888])
print("""messages to array:""")
print(messages.toarray()[:10])
v: dict[str, int] = cv.vocabulary_
print(f"vocabulary map: {dict(list(v.items())[:10])}")
# first index
v_first = [key for key, value in v.items() if value == 0]
print(f"first index: {v_first}")
# 888th index
v_888 = [key for key, value in v.items() if value == 888]
print(f"888th index: {v_888}")

# ----
# cv = CountVectorizer(max_features=6)
# documents = [
#     "Hello world. Today is amazing. Hello hello",
#     "Hello mars, today is perfect"
# ]
# cv.fit(documents)
# print(cv.get_feature_names_out())
# # use transform to create document-term matrix
# out = cv.transform(documents)
# print(out.todense())
