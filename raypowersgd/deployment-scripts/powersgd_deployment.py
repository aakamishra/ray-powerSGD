import wandb
import os
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
from torchvision.models import resnet
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import ray

import torch
from powersgd import PowerSGD, RankScaler, optimizer_step, Config


def rtrain(model, train_loader, optimizer, powersgd, epoch, criterion, rankscaler):
    """
    Function for running gradient batched - compressed training cycle
    """
    start = time.time_ns()
    model.train()
    """
    train_loader = trainset_shard.iter_torch_batches(
            batch_size=batch_size,
        )
    """
    for batch_idx, (image, label) in enumerate(train_loader):
        data, target = image, label
        
        model_pass_start_time = time.time_ns()
        output = model(data)
        loss = criterion(output, target)
        model_total_time = time.time_ns() - model_pass_start_time

        with model.no_sync():
            loss.backward()
        
        optimizer_step_start = time.time_ns()
        optimizer_step(optimizer, powersgd)
        optimizer_step_time = time.time_ns() - optimizer_step_start
        
        net_gpu_ratio = optimizer_step_time / model_total_time
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"train/epoch": epoch, 
                       "train/batch_idx": batch_idx, 
                       "train/loss": loss.item(),
                       "train/model_total_time": model_total_time,
                       "train/optimizer_step_time": optimizer_step_time,
                       "train/net_gpu_ratio": net_gpu_ratio
                       })
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


    # include trainset set shard
    trainset_shard = session.get_dataset_shard("train")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=worker_batch_size,
                                            shuffle=False, num_workers=2)

    train_loader = train.torch.prepare_data_loader(train_loader)
    test_loader = train.torch.prepare_data_loader(test_loader)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize the model using the legacy API
    model = resnet.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 10)
    model = train.torch.prepare_model(model)

    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    rankscaler = RankScaler()
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)    
    powersgd = PowerSGD(list(params), config=Config(
        rank=2,  # lower rank => more aggressive compression
        min_compression_rate=10,  # don't compress gradients with less compression
        num_iters_per_step=2,  #   # lower number => more aggressive compression
        start_compressing_after_num_steps=0,
    ))

    accuracy_results = []
    os.environ["WANDB_API_KEY"] = "<SET YOUR WANDB KEY HERE>"
    wandb.init(project="iolaus/powersgd-resnet-v2-trial-10")
    for epoch in range(epochs):
        
        start_time = time.time_ns()
        rtrain(model, train_loader, optimizer, powersgd, epoch, criterion, rankscaler)
        stop_time = time.time_ns() - start_time
        accuracy = rtest(model, test_loader)
        checkpoint = TorchCheckpoint.from_state_dict(model.module.state_dict())
        metrics = {"accuracy": accuracy, 'epoch': epoch, "time": stop_time}
        wandb.log(metrics)
        session.report(metrics, checkpoint=checkpoint)
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

    # create shardable dataset
    train_dataset: ray.data.Dataset = ray.data.read_datasource(
        SimpleTorchDatasource(), dataset_factory=get_train_dataset
    )
    train_dataset = train_dataset.map_batches(convert_batch_to_numpy)
    print("Batches Converted")
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr": 1e-4,
            "batch_size": 128,
            "epochs": 100
        },
        scaling_config=scaling_config,
        datasets={"train": train_dataset},
        run_config= RunConfig(
            callbacks=[]
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

    ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
    print("finished ray init")
    accs = train_resnet50_cifar(num_workers=args.num_workers, use_gpu=args.use_gpu)
    print(accs)
    
    