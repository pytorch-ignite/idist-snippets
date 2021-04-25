import argparse
import time
import torch
from torch.optim import SGD
from torch.utils.data import Dataset
from torchvision.models import wide_resnet50_2
import ignite.distributed as idist
from ignite.engine import Engine


class MockDataset(Dataset):
    def __init__(self, nb_samples=128, labels=100):
        self._labels = labels
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y

def _mp_fn(local_rank):
    # A training step printing process information and underlying data
    def _train_step(e, batch):
        print(
            f"Process {idist.get_rank()}/{idist.get_world_size()} : Epoch {e.state.epoch} - {e.state.iteration} : batch={batch}")
        # This is a synchronization point where we are waiting all the process to finish the previous commands
        idist.barrier()
        if idist.get_rank() == 0:
            time.sleep(2)

    # Define dummy input data for sake of simplicity
    batch_data = [0, 1, 2]
    # Running the _train_step function on whole batch_data iterable only once
    trainer = Engine(_train_step)
    trainer.run(batch_data, max_epochs=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training-Step")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)
    args = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributed.launch, horovodrun, slurm)
    kwargs = dict()
    if args.nproc_per_node is not None:
        kwargs["nproc_per_node"] = args.nproc_per_node
    with idist.Parallel(backend=args.backend, **kwargs) as parallel:
        parallel.run(_mp_fn)
