import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

# mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=ToTensor())
# print(mnist_train[0])

model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 10))
model.load_state_dict(torch.load("mnist-model.pth", weights_only=True))

img = Image.open("3.png")
img.thumbnail((28, 28))
img = img.convert("L")

t = ToTensor()
X = 1 - t(img).reshape((-1, 784))
print(X)
print(X.shape)
outputs = model(X)
print(nn.functional.softmax(outputs, dim=1))
