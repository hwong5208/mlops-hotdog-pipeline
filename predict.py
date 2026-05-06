import sys
import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

CLASSES = ["hot_dog", "not_hot_dog"]
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hotdog-not-hotdog"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latest_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Train the model first.")
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from run: {run_id}")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()
    return model


def predict(model, image_path: str):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    predicted = probs.argmax().item()
    confidence = probs[predicted].item()
    return CLASSES[predicted], confidence, probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/test/hot_dog/1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    model = load_latest_model()
    label, confidence, probs = predict(model, image_path)

    print(f"\nImage   : {image_path}")
    print(f"Result  : {label.upper().replace('_', ' ')}")
    print(f"Confidence: {confidence:.1%}")
    print(f"  hot_dog     : {probs[0]:.1%}")
    print(f"  not_hot_dog : {probs[1]:.1%}")


if __name__ == "__main__":
    main()
