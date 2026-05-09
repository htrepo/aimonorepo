import pathlib

import torch
from torch import nn

current_dir = pathlib.Path(__file__).parent

X_mean = torch.load(current_dir / "model" / "X_mean.pt", weights_only=True)
X_std = torch.load(current_dir / "model" / "X_std.pt", weights_only=True)
y_mean = torch.load(current_dir / "model" / "y_mean.pt", weights_only=True)
y_std = torch.load(current_dir / "model" / "y_std.pt", weights_only=True)

model = nn.Linear(2, 1)
model.load_state_dict(torch.load(current_dir / "model" / "model.pt", weights_only=True))
model.eval()


with torch.no_grad():
    X_data = torch.tensor([[1, 10000], [2, 10000], [3, 20000]], dtype=torch.float32)
    X_data = torch.tensor([[2, 10000], [2, 20000], [3, 30000]], dtype=torch.float32)

    # normalize unseen data as model is trained on normalized data
    prediction = model((X_data - X_mean) / X_std)
    # denormalize the prediction
    print(prediction * y_std + y_mean)
