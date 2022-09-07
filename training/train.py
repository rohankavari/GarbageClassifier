import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data import train_dataloader
from model import GarbageClassifier

dataiter = iter(train_dataloader)
images, labels = dataiter.next()
# print(images[2].size())

EPOCHS = 10
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GarbageClassifier().to(device)
optimizer = Adam(model.parameters(), lr=LR,)
criterion = nn.CrossEntropyLoss()
print(f"training starting on {device}")
for epoch in range(EPOCHS):
    step = 0
    running_loss = []
    st = time.perf_counter()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred_labels = model(images)
        loss = criterion(pred_labels, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        step += 1
    print(
        f"Epoch:{epoch}, loss:{round(np.mean(running_loss),3)},time:{time.perf_counter()-st} s")

torch.save(model.state_dict(),
           f"./training/models/final-at-{round(np.mean(running_loss),3)}.pt")
