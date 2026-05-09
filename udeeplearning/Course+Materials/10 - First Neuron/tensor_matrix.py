import torch

X = torch.tensor([[10], [38], [100], [150]])
# print(X.size(1))
print(f"X: {X}")
print(f"X[2, 0]: {X[2, 0]}")  # 2nd row, 1st column
print("all rows, 1st column:", X[:, 0])  # all rows , 1st column
print(f"X.shape: {X.shape}")
print(f"X.size(): {X.size()}")
print(f"X.ndim: {X.ndim}")
