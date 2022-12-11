import ray
import cupy as cp
import torch
import ray.util.collective as collective

"""Reworked example of all-reduce communication between intialized Ray Nodes"""

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):

        # log GPU information for Azure cluster deployment
        print(ray.get_gpu_ids())
        print(torch.cuda.is_available())

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get device type
        print(device)

        # sample torch values
        self.send = torch.zeros([2, 4], device=device, dtype=torch.int32)
        self.recv = torch.zeros([2, 4], device=device, dtype=torch.int32)

    def setup(self, world_size, rank):
        collective.init_collective_group(world_size, rank, "nccl", "default")
        return True

    def compute(self):
        collective.allreduce(self.send, "default")
        return self.send

    def destroy(self):
        collective.destroy_group()


if __name__ == "__main__":
    # intialize ray with two gpus as resources
    ray.init(num_gpus=2)

    num_workers = 2
    workers = []
    init_rets = []
    # create workers, each with one gpu assigned
    for i in range(num_workers):
        w = Worker.remote()
        workers.append(w)
        init_rets.append(w.setup.remote(num_workers, i))
    
    # process init_rets from worker intialization
    rets = ray.get(init_rets)
    print(rets)

    # get the results of the arrays transmitted with the all-reduce process
    results = ray.get([w.compute.remote() for w in workers])
    print(results)
    ray.shutdown()