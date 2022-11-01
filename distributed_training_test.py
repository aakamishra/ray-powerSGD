import torch
import ray.train as train
from ray.train.trainer import Trainer
from ray.train.callbacks import JsonLoggerCallback
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Dict
import argparse
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ray.air import session
import time


from powersgd import PowerSGD, Config, optimizer_step



def rtrain(model, train_loader, optimizer, powersgd, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        output = model(data)
        loss = criterion(output, target)

        start = time.time_ns()
        with model.no_sync():
            loss.backward()
        optimizer_step(optimizer, powersgd)
        if batch_idx % 100 == 0:
            print('minibatch time: ', time.time_ns() - start)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def rtest(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (data, target) in test_loader:
            images, labels = data, target
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return (100 * correct // total)


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // train.world_size()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    train_loader = train.torch.prepare_data_loader(train_loader)
    test_loader = train.torch.prepare_data_loader(test_loader)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = torchvision.models.resnet50(pretrained=True)
    model = train.torch.prepare_model(model)

    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    powersgd = PowerSGD(list(params), config=Config(
        rank=1,  # lower rank => more aggressive compression
        min_compression_rate=10,  # don't compress gradients with less compression
        num_iters_per_step=2,  #   # lower number => more aggressive compression
        start_compressing_after_num_steps=0,
    ))

    accuracy_results = []

    for epoch in range(epochs):
        
        rtrain(model, train_loader, optimizer, powersgd, epoch, criterion)
        accuracy = rtest(model, test_loader)
        train.report(accuracy=accuracy)
        accuracy_results.append(accuracy)

    return accuracy_results


def train_resnet50_cifar(num_workers=4, use_gpu=True):
    trainer = Trainer(
        backend="torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    result = trainer.run(
        train_func=train_func,
        config={
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 10
        },
        callbacks=[JsonLoggerCallback()])
    trainer.shutdown()
    print(f"Loss results: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=4,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Enables GPU training")

    args, _ = parser.parse_known_args()
    print("args: ", args)

    import ray
    ray.init(address=args.address)
    accs = train_resnet50_cifar(num_workers=args.num_workers, use_gpu=args.use_gpu)
    print(accs)