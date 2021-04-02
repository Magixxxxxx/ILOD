import torch
from torchvision import datasets,transforms

train_dataset = datasets.MNIST(
    root = '../data/',
    train = True,
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
)



train_loader = iter(torch.utils.data.DataLoader(train_dataset))

print(train_loader.next()[1])
print(train_loader.next()[1])
print(train_loader.next()[1])

train_dataset = datasets.CIFAR10(
    root = '../data/',
    train = True,
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
    )

train_loader = iter(torch.utils.data.DataLoader(train_dataset))

print(train_loader.next()[1])
print(train_loader.next()[1])
print(train_loader.next()[1])
print(train_loader.next()[1])
