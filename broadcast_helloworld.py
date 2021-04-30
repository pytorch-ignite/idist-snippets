import argparse

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.engine import Events


def training(local_rank):
    # Broadcast string s1 from rank 0 to all processes
    engine = Engine(lambda e, batch: 1)

    @engine.on(Events.EPOCH_STARTED)
    def _(engine):
        s1 = ""
        if idist.get_rank() == 3:
            s1 = "Hello world"
        print({s1})
        s1 = idist.broadcast(s1, src=3)
        print(f"data {s1} process {idist.get_rank()}/{idist.get_world_size()} ")

    num_epochs = 1
    num_iters = 1
    data = range(num_iters)
    engine.run(data, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hello-World")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributed.launch, horovodrun, slurm)
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training)
