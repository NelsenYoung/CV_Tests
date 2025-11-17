# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("residentmario/wheres-waldo")

# print("Path to dataset files:", path)


# from PIL import Image
# import matplotlib.pyplot as plt

# img_path = path + "/wheres-waldo/Hey-Waldo/256/waldo/10_3_1.jpg"
# img = Image.open(img_path)

# plt.imshow(img)
# plt.axis("off")
# plt.show()

import math
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

path = "/Users/nelsenyoung/.cache/kagglehub/datasets/residentmario/wheres-waldo/versions/2"

dataset = datasets.ImageFolder(
    path + '/wheres-waldo/Hey-Waldo/64', 
    transform = ToTensor()
    )

print(len(dataset))
img, label = dataset[0]
print(img.shape, label)

# Source - https://stackoverflow.com/a
# Posted by Fábio Perez, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-17, License - CC BY-SA 4.0

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
batch_size = 32

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linera_relu_stack = nn.Sequential(
            nn.Linear(3 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(1, -1)
        return torch.squeeze(self.linera_relu_stack(x))

model = Network()
print(model)

lr = 0.001
epochs = 10
loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        y = y.float()
        pred = model(X)
        # print(pred, y)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_labels = (pred >= 0.5).float()
            correct += (pred_labels == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# torch.save(model.state_dict(), "ww_bc_model.pth")
# print("Saved PyTorch Model State to model.pth")