import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def model_fn(model_dir):
    path = Path(model_dir)
    classes_file = path / "classes.json"
    labels = json.loads(classes_file.read_text()) if classes_file.exists() else ["hotdog", "not_hotdog"]
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(labels))
    net.load_state_dict(torch.load(path / "model.pth", map_location="cpu"))
    net.eval()
    net.labels = labels
    return net


def input_fn(data, content_type):
    if content_type in ("image/jpeg", "image/png", "application/x-image"):
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(image, model):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return {"class": model.labels[idx], "confidence": round(probs[idx].item(), 4)}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
