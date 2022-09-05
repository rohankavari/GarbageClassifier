import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from model import GarbageClassifier

transform = transforms.Compose([
    transforms.ToTensor()
])
data_dir = "./archive/Garbage classification/Garbage classification"
dataset = ImageFolder(root=data_dir, transform=transform)

train_len = int(0.8*len(dataset))
test_len = int(0.1*len(dataset))
val_len = len(dataset)-train_len-test_len

train_dataset, test_dataset, val_dataset = random_split(
    dataset, [train_len, test_len, val_len])

BATCH_SIZE = 32

train_dataloader, test_dataloader, val_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True)

dataiter = iter(train_dataloader)
images, labels = dataiter.next()
# print(images[2].size())

EPOCHS = 1
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GarbageClassifier().to(device)
optimizer = Adam(model.parameters(), lr=LR,)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    step = 0
    running_loss = []
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred_labels = model(images)
        loss = criterion(pred_labels, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        step += 1
    print(f"Epoch:{epoch}, loss:{round(np.mean(running_loss),3)}")

# torch.save(model.state_dict(), f"./models/{}")
