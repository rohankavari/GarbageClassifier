from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

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

if __name__ == "__main__":
    a = iter(train_dataloader)
    # print(a.shape)
    im, la = a.next()
    print(la)
    print(dataset.classes)
    print(dataset.class_to_idx)
