from PIL import Image
import torch
from torchvision import models, transforms
import io

MODEL_PATH = "checkpoint_epoch4.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["classes"]
num_classes = len(class_names)

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def predict_disease(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        idx = output.argmax(1).item()
        confidence = torch.softmax(output,1)[0][idx].item()

    return class_names[idx], confidence