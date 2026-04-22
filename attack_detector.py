import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class SimpleDetector(nn.Module):
    def __init__(self):
        super(SimpleDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(224*224*3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # normal / attacked
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleDetector()

def detect_attack(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.fromarray(image)
    x = transform(img).view(1, -1)

    output = model(x)
    pred = torch.argmax(output)

    return "Normal" if pred.item() == 0 else "Attack"