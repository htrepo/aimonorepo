import gradio as gr
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

resnet50_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
resnet50_model.fc = nn.Identity()
resnet50_model = resnet50_model.to(device)

fc_model = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
fc_state_dict = torch.load("model.pth", weights_only=True)
fc_model.load_state_dict(fc_state_dict)
fc_model = fc_model.to(device)

model = nn.Sequential(resnet50_model, fc_model)
model = model.to(device)
model.eval()

# tire = Image.open("./prediction/tire3.jpg")
# tire_tensor = preprocess(tire)
# tire_tensor = tire_tensor.unsqueeze(dim=0)
# tire_tensor = tire_tensor.to(device)
# with torch.no_grad():
#    y_pred = torch.sigmoid(model(tire_tensor))
#    print(y_pred)
#    pass


def predict_image(image_pixels):
    tire = Image.fromarray(image_pixels)
    tire_tensor = preprocess(tire)
    tire_tensor = tire_tensor.unsqueeze(dim=0)
    tire_tensor = tire_tensor.to(device)
    with torch.no_grad():
        y_pred = torch.sigmoid(model(tire_tensor))
        y_pred_value = y_pred.item()
        percentage = round(y_pred_value * 100, 3)

        return f"By {percentage}%, this is a good tire"


demo = gr.Interface(fn=predict_image, inputs=[gr.Image()], outputs=gr.Text())
demo.launch(share=True)
