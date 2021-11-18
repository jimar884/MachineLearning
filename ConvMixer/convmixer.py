import enum
import torch
from torch import optim
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = 32
BATCH_SIZE = 128
NUM_EPOCHS = 300
LR = 0.05

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim=256, depth=16, kernel_size=8, patch_size=1, n_class=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_class)
    )


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Set device")

    transform_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, .5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, .5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        '../data', train=True, download=False, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = torchvision.datasets.CIFAR10(
        '../data', train=False, download=False, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    print("Set trainloader and testloader")

    net = ConvMixer(dim=256, depth=16, kernel_size=8, patch_size=1, n_class=10)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=LR)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    print("Set net, criterion, optimizer, and schedular")

    print("Start training")
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0
    for i in range(NUM_EPOCHS):
        # train
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for j, (input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
        train_loss = train_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print('epoch: %4d | train | loss: %.8f | acc: %.8f' % (i+1, train_loss, train_accuracy))

        # test
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for j, (input, label) in enumerate(testloader):
                input, label = input.to(device), label.to(device)
                optimizer.zero_grad()
                output = net(input)
                loss = criterion(output, label)

                test_loss += loss.item()
                _, predict = output.max(1)
                total += label.size(0)
                correct += predict.eq(label).sum().item()
        test_loss = test_loss / total
        test_accuracy = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print('epoch: %4d | test  | loss: %.8f | acc: %.8f' % (i+1, test_loss, test_accuracy))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net.state_dict(), 'ConvMixer300.pth')
    print("Finish training")
    print('best accuracy: %.8f' % (best_accuracy))

    fig_loss = plt.figure(figsize=(10,5))
    plt.title("Loss")
    plt.plot(train_losses,label="train")
    plt.plot(test_losses,label="test")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    fig_loss.savefig("fig_loss_ConvMixer300.png")

    fig_acc = plt.figure(figsize=(10,5))
    plt.title("Loss")
    plt.plot(train_accuracies,label="train")
    plt.plot(test_accuracies,label="test")
    plt.xlabel("epochs")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()
    fig_acc.savefig("fig_acc_ConvMixer300.png")

    print("Save loss and acc")

