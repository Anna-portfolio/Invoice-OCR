import torch
sd = torch.load("models/document_classifier.pt")
print(sd.keys())
