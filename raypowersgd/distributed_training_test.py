import torch
import ray.train as train
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.data.datasource import SimpleTorchDatasource
from ray.air.config import ScalingConfig, RunConfig
from ray.train.torch import TorchCheckpoint
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
from ray.air import session, Checkpoint
import time
import numpy as np
import ray


from powersgd import PowerSGD, Config, optimizer_step



def rtrain(model, train_loader, optimizer, powersgd, epoch, criterion):
    """
    Function for running gradient batched - compressed training cycle
    """
    start = time.time_ns()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        output = model(data)
        loss = criterion(output, target)

        with model.no_sync():
            loss.backward()
        optimizer_step(optimizer, powersgd)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('Epoch time: ', time.time_ns() - start)

def rtest(model, test_loader):
    """
    simple test function for cifar accuracy counts
    """
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
    """
    Distributed worker function for ray trainer loop
    """

    # load config values
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // train.world_size()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset_shard = session.get_dataset_shard("train")
    
    train_loader = trainset_shard.iter_torch_batches(
            batch_size=config["batch_size"],
        )
    #train_loader = torch.utils.data.DataLoader(trainset_shard, batch_size=batch_size,
    #                                        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=worker_batch_size,
                                            shuffle=False, num_workers=2)

    #train_loader = train.torch.prepare_data_loader(train_loader)
    test_loader = train.torch.prepare_data_loader(test_loader)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = torchvision.models.resnet50(pretrained=False)
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
        checkpoint = TorchCheckpoint.from_state_dict(model.module.state_dict())
        session.report(accuracy=accuracy, checkpoint=checkpoint)
        accuracy_results.append(accuracy)

    return accuracy_results


def convert_batch_to_numpy(batch):
    images = np.array([image.numpy() for image, _ in batch])
    labels = np.array([label for _, label in batch])
    return {"image": images, "label": labels}


def get_train_dataset():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    return trainset

def train_resnet50_cifar(num_workers=4, use_gpu=True):

    train_dataset: ray.data.Dataset = ray.data.read_datasource(
        SimpleTorchDatasource(), dataset_factory=get_train_dataset
    )
    train_dataset = train_dataset.map_batches(convert_batch_to_numpy)
    print("Batches Converted")
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 10
        },
        scaling_config=scaling_config,
        datasets={"train": train_dataset},
        run_config= RunConfig(
            callbacks=[WandbLoggerCallback(
            project="Gradient_Compression_Project",
            api_key_file="wandb_key.txt",
            log_config=True)]
            )
        )
    
    result = trainer.fit()
    print(f"Last result: {result.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    """
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Enables GPU training")

    args, _ = parser.parse_known_args()
    print("args: ", args)

    #ray.init(address=args.address, ignore_reinit_error=True)
    ray.init(address=ray.services.get_node_ip_address() + ":6379", ignore_reinit_error=True, include_dashboard=False)
    print("finished ray init")
    accs = train_resnet50_cifar(num_workers=args.num_workers, use_gpu=args.use_gpu)
    print(accs)