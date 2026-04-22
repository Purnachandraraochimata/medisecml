import torch
import torch.nn as nn
import numpy as np

class KeyGenerator(nn.Module):
    def __init__(self):
        super(KeyGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 224*224*3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

# Generate key
def generate_ml_key(shape):
    model = KeyGenerator()
    noise = torch.randn(1, 100)
    key = model(noise).detach().numpy()

    key = key.reshape(shape)
    key = ((key + 1) * 127.5).astype(np.uint8)  # scale to 0–255

    return key