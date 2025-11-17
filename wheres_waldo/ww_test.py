import math
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from wheres_waldo_network import Network, test_dataset

device = "cpu"

model = Network().to(device)
model.load_state_dict(torch.load("ww_bc_model.pth", weights_only=True))

classes = [
    'not waldo',
    'waldo'
]

model.eval()
for x, y in test_dataset:
    if y == 0:
        continue
    print(x.shape, y)
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        print(pred.item())
        predicted, actual = classes[round(pred.item())], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')