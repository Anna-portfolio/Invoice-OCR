# create_dummy_pt.py
import torch
from src.classification.inference import SimpleCNN
import os

os.makedirs("models", exist_ok=True)
model = SimpleCNN(num_classes=5)
torch.save(model.state_dict(), "models/document_classifier.pt")
print("Dummy model saved correctly under models")
