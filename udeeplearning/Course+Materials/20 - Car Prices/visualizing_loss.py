import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

# Pandas: Reading the data
current_dir = pathlib.Path(__file__).parent
df = pd.read_csv(current_dir / "data" / "used_cars.csv")

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

# Torch: Creating X and y data (as tensors)
X = torch.column_stack([torch.tensor(age, dtype=torch.float32), torch.tensor(milage, dtype=torch.float32)])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std
# sys.exit()

LEARNING_RATE = 0.01
model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

losses = []
for i in range(0, 200):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    # if i % 100 == 0:
    #    print(loss.item())
    if i % 20 == 0:
        # print bias and weight for 2 parameters
        print(f"weight: {model.weight}")
        print(f"bias: {model.bias}")
# print(losses)
plt.plot(losses)
plt.show()

X_data = torch.tensor([[1, 10000], [2, 10000], [3, 20000]], dtype=torch.float32)
X_data = torch.tensor([[2, 10000], [2, 20000], [3, 30000]], dtype=torch.float32)

# normalize unseen data as model is trained on normalized data
prediction = model((X_data - X_mean) / X_std)
# denormalize the prediction
print(prediction * y_std + y_mean)
