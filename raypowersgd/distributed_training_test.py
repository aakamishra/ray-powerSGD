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
import torch.nn.functional as F
from ray.air import session, Checkpoint
import time
import numpy as np
import ray
from torch.optim import lr_scheduler


from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, NamedTuple, Union
from types import SimpleNamespace
import torch


"""243 Group implementation of PowerSGD"""
# Original code implementation can be found here: https://github.com/epfml/powersgd



############################# UTLITIES ###################################

def orthogonalize(matrix: torch.Tensor, eps=torch.tensor(1e-16)):
    if matrix.shape[-1] == 1:
        matrix.div_(torch.maximum(matrix.norm(), eps))
    else:
        matrix.copy_(torch.linalg.qr(matrix).Q)


def pack(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def params_in_optimizer(optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()  # type: ignore


def flatten(tensors: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out


########################## ALL-REDUCE PyTorch Integration ###############################

def allreduce_average(data, *args, **kwargs):
    """All-reduce average if torch.distributed is available, otherwise do nothing"""
    if is_distributed():
        data.div_(torch.distributed.get_world_size())  # type: ignore
        #print("pytorch world size: ", torch.distributed.get_world_size())        
        start_time = time.time_ns()
        ret = torch.distributed.all_reduce(data, *args, **kwargs)  # type: ignore
        wandb.log({"Communication Bits": 8 * data.nelement() * data.element_size(), "All Reduce Time": time.time_ns() - start_time})
        return ret
        
    else:
        return SimpleNamespace(wait=lambda: None)

class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass


class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        buffer, shapes = pack(gradients)
        allreduce_average(buffer)
        out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100


############################## PowerSGD Main ###################################


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate(gradients)

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate(compressed_grads),
            self._allreduce.aggregate(uncompressed_grads),
        )

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """
        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])), 
                    in_batch, 
                    out=out_batch
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )

            # Average across workers
            if is_distributed():
                num_workers = torch.distributed.get_world_size()
                torch.distributed.all_reduce(out_buffer)
            else:
                num_workers = 1

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    alpha=1/num_workers
                )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Increment the step counter
        self.step_counter += 1

        return output_tensors

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats



def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)

def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = params_in_optimizer(optimizer)
    grads = [p.grad.data for p in params]  # type: ignore
    avg_grads = aggregator.aggregate(grads)  # subtracts the approximation from grads

    # Temporarily set parameter's gradients to the aggregated values
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, grads):
        p.grad = g


def rtrain(model, train_loader, optimizer, powersgd, epoch, criterion):
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
        optimizer.zero_grad()
        model_pass_start_time = time.time_ns()
        output = model(data)
        loss = criterion(output, target)
        model_total_time = time.time_ns() - model_pass_start_time

        loss.backward()
        
        optimizer_step_start = time.time_ns()
        #optimizer_step(optimizer, powersgd)
        optimizer.step()
        optimizer_step_time = time.time_ns() - optimizer_step_start
        
        # net_gpu_ratio = wandb.run.summary["All Reduce Time"] / model_total_time
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"train/epoch": epoch, 
                       "train/batch_idx": batch_idx, 
                       "train/loss": loss.item(),
                       "train/model_total_time": model_total_time,
                       "train/optimizer_step_time": optimizer_step_time,
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

    # trainset_shard = session.get_dataset_shard("train")
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
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)    
    powersgd = PowerSGD(list(params), config=Config(
        rank=2,  # lower rank => more aggressive compression
        min_compression_rate=10,  # don't compress gradients with less compression
        num_iters_per_step=2,  #   # lower number => more aggressive compression
        start_compressing_after_num_steps=0,
    ))
    
    
    accuracy_results = []
    os.environ["WANDB_API_KEY"] = "8f7086db96f9edfde9aae91cfcf98f1f445333f5"
    wandb.init(project="powersgd-resnet-v2-trial-14-10gbps-nopowersgd")
    for epoch in range(epochs):
        
        start_time = time.time_ns()
        rtrain(model, train_loader, optimizer, powersgd, epoch, criterion)
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

    #train_dataset: ray.data.Dataset = ray.data.read_datasource(
     #   SimpleTorchDatasource(), dataset_factory=get_train_dataset
    #)
    #train_dataset = train_dataset.map_batches(convert_batch_to_numpy)
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
        #datasets={"train": train_dataset},
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

    #ray.init(address=args.address, ignore_reinit_error=True)
    ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
    print("finished ray init")
    accs = train_resnet50_cifar(num_workers=args.num_workers, use_gpu=args.use_gpu)
    print(accs)
