import torch.nn as nn
import torch
import torch.nn.functional as F


class GarbageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(186000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    model = GarbageClassifier()
    model.load_state_dict(torch.load("./training/models/final-at-1.726.pt",map_location=torch.device('cpu')))

    a = torch.rand((1, 3, 384, 512))
    print(model(a))

    img = Image.open("/home/conmove/Projects/homeprj/GarbageClassifier/archive/Garbage classification/Garbage classification/metal/metal1.jpg")
    convert_tensor = transforms.ToTensor()
    print(convert_tensor(img).shape)
    b=torch.unsqueeze(convert_tensor(img), dim=0)
    print(b.shape)
    # print(model(a))
    print(model(b))
