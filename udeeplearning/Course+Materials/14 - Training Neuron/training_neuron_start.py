import torch
from torch import nn

X1 = torch.tensor([[10.0]])  # Input: Temperature in °C
y1 = torch.tensor([[50.0]])  # Actual value: Temperature °F

X2 = torch.tensor([[37.78]])  # Input: Temperature in °C
y2 = torch.tensor([[100.0]])  # Actual value: Temperature °F

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

for i in range(0, 50000):
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(model.weight)
        print(model.bias)
