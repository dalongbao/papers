import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from pathlib import Path

checkpoint_dir = Path('./ckpts')
checkpoint_dir.mkdir(exist_ok=True)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                       std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                       std=[0.2675, 0.2565, 0.2761])
])

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    """
    Save model checkpoint with all training state.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance 
        epoch (int): Current epoch number
        loss (float): Current loss value
        accuracy (float): Current accuracy value
        filename (str): Name of checkpoint file
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'config': model.config
    }
    
    save_path = Path(checkpoint_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: Model instance to restore
        optimizer: Optimizer instance to restore
        filename (str): Name of checkpoint file
        
    Returns:
        tuple: (epoch, loss, accuracy)
    """
    load_path = Path(checkpoint_dir) / filename
    if not load_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
        
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']

def get_dataset(dataset='cifar10'):
    """
    Get training and test datasets.
    
    Args:
        dataset (str): Name of dataset ('cifar10', 'cifar100', 'mnist', 
                      'fashionmnist', 'imagenet')
    
    Returns:
        tuple: (training_dataset, test_dataset)
        
    Raises:
        ValueError: If dataset is not supported
    """
    dataset_map = {
        'cifar10': (datasets.CIFAR10, 10),
        'cifar100': (datasets.CIFAR100, 100),
        'mnist': (datasets.MNIST, 10),
        'fashionmnist': datasets.FashionMNIST,
        'imagenet': datasets.ImageNet
    }
    
    dataset = dataset.lower()
    data_dir = Path("../data") / dataset
    if dataset not in dataset_map:
        raise ValueError(f"Dataset {dataset} not supported. Choose from {list(dataset_map.keys())}")
    
    if dataset != "imagenet":
        DatasetClass, num_classes = dataset_map[dataset]
        
        training_data = DatasetClass(
            root=str(data_dir),
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_data = DatasetClass(
            root=str(data_dir),
            train=False,
            download=True,
            transform=test_transform
        )

    else:
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        training_data = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_data = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        num_classes = 1000
    
    return training_data, test_data, num_classes
