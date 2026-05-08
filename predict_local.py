import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "model.pth"
LABELS = ["hotdog", "not_hotdog"]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(image_path):
    model = load_model(MODEL_PATH)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()
    print(f"Prediction : {LABELS[pred]}")
    print(f"Confidence : {probs[pred]:.1%}")
    print(f"  hotdog   : {probs[0]:.1%}")
    print(f"  not_hotdog: {probs[1]:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_local.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
