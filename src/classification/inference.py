import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


# SimpleCNN
class SimpleCNN(nn.Module):
    """
    Simple CNN for document classification.
    Input: 1x256x256
    Two poolings to match fc1 input size.
    Output: num_classes
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # po dwóch poolings: 256 -> 128 -> 64
        self.fc1 = nn.Linear(32*64*64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)  # drugi pooling
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DocumentPredictor
class DocumentPredictor:
    def __init__(self, model_path, classes=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # default classes if none provided
        self.classes = classes or ["id_card", "contract", "invoice", "vehicle_document", "other"]

        # create model and load state_dict
        self.model = SimpleCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # transform for input image
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_path):
        # open image and apply transforms
        img = Image.open(image_path).convert("L")
        x = self.transform(img).unsqueeze(0).to(self.device)  # add batch dim
        with torch.no_grad():
            logits = self.model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]


# Convenience function
def classify_document(image_path, model_path, classes=None):
    predictor = DocumentPredictor(model_path, classes)
    return predictor.predict(image_path)
