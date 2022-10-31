import ray
import ray.util.collective as col
import torch


@ray.remote(num_cpus=1)
class GPUWorker:
    def __init__(self):
        self.gradients = torch.ones((10,), dtype=torch.float32)

    def setup(self, world_size, rank):
       col.init_collective_group(
           world_size=world_size, 
           rank=rank, 
           backend="nccl")

    def allreduce(self):
        col.allreduce(self.gradients)
        return self.gradients


class AllReduceServer:
    def __init__(self, num_workers=4, verbose=1):
        self.num_workers = num_workers
        self.verbose = verbose
        self.workers = [GPUWorker.remote() for i in range(self.num_workers)]

    def setup(self):
        setup_rets = ray.get([w.setup(16, i) for i, w in enumerate(self.workers)])
        if self.verbose:
            print(setup_rets)

    def allreduce_sync(self):
        results = ray.get([w.allreduce.remote() for w in self.workers])
        return results


    def run(self):
        results = self.allreduce_sync()
        print(results)
        

if __name__ == "__main__":
    server = AllReduceServer(num_workers=4)
    server.setup()
    server.run()
