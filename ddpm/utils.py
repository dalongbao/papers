import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def get_data(
    batch_size: int = 512, 
    num_workers: int = 2,
):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return trainloader, testloader


def eval(model, testloader):
    device = get_device()
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = F.cross_entropy(pred, y)

            total_loss += loss.item() * y.size(0)
            total_acc += (pred.argmax(1) == y).float().sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_acc / total_samples

def get_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
