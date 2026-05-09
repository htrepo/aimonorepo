import pathlib
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn

start_time = time.time()

current_dir = pathlib.Path(__file__).parent
df = pd.read_csv(current_dir / "data" / "SMSSpamCollection", sep="\t", names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

model = nn.Linear(1000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

loss_list = []
for i in range(0, 100000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if i % 10000 == 0:
        print(loss)

plt.plot(range(0, 10000), loss_list)
plt.show()


model.eval()
with torch.no_grad():
    y_pred = nn.functional.sigmoid(model(X))
    print(y_pred)
    print(y_pred.min())
    print(y_pred.max())

end_time = time.time()
print(f"Total time: {end_time - start_time}")
