import torch

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1

print(f"b.shape is {b.shape}")
print(f"b.size is {b.size()}")
print(f"w1.shape is {w1.shape}")
print(f"X1.shape is {X1.shape}")
print(f"b.size() is {b.size()}")
print(f"X1.size() is {X1.size()}")
print(f"value: y_pred[1].item() is {y_pred[1].item()}")
print(f"y_pred is {y_pred}")
print(f"y_pred.ndim is {y_pred.ndim}")
print(f"b.ndim is {b.ndim}")
print(f"w1.ndim is {w1.ndim}")
print(f"X1.ndim is {X1.ndim}")
