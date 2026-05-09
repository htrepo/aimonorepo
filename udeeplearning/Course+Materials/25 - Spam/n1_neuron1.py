import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn

current_dir = pathlib.Path(__file__).parent
df = pd.read_csv(current_dir / "data" / "SMSSpamCollection", sep="\t", names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

model = nn.Linear(1000, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

loss_list = []
for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if i % 1000 == 0:
        print(loss)

# plot the loss

plt.plot(loss_list)
plt.show()


#####################################
# finding issue...
# sigmoid activation function not added - we will in another code
#################################################

model.eval()
with torch.no_grad():
    y_pred = model(X)
    print(y_pred)
    print(y_pred.min())
    print(y_pred.max())

# output has -ve values so something is wrong with this code.
# tensor([[-0.0601],
#         [-0.0206],
#         [ 0.8279],
#         ...,
#         [-0.0666],
#         [ 0.1580],
#         [ 0.1043]])
# tensor(-0.7051)
# tensor(1.5195)
