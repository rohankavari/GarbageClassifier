import cv2
import torch
from torchvision import transforms

from training.model import GarbageClassifier


async def predict(img):
    try:
        model = GarbageClassifier()
        model.load_state_dict(torch.load(
            "./training/models/final-at-1.726.pt", map_location=torch.device('cpu')))

        convert_tensor = transforms.ToTensor()
        img = torch.unsqueeze(convert_tensor(img), dim=0)
        print(model(img))
        res = model(img)
        return res
    except Exception as e:
        print(str(e))
