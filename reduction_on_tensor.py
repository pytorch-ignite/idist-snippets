import argparse

import ignite.distributed as idist
import torch
from ignite.engine import Engine
from ignite.engine import Events


def training(local_rank):
    # Broadcast string s1 from rank 0 to all processes
    engine = Engine(lambda e, batch: 1)

    @engine.on(Events.EPOCH_STARTED)
    def _(engine):
        tensor = torch.ones(1)
        tensor = idist.all_reduce(tensor)  # or use idist.all_reduce(tensor)
        print(f"data {tensor} process {idist.get_rank()}/{idist.get_world_size()} ")

    num_epochs = 1
    num_iters = 1
    data = range(num_iters)
    engine.run(data, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hello-World")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (slurm, torch.distributed.launch, horovodrun)
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training)
