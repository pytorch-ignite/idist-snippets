import argparse

import horovod.torch as hvd
import torch
from horovod import run
from torch.nn import NLLLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import wide_resnet50_2


class RndDataset(Dataset):
    def __init__(self, nb_samples=128):
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y


def _mp_train(world_size, backend, config):
    # Specific hvd
    hvd.init()
    print({hvd.local_rank()}, ": run with config:", config, " - backend=", backend)

    device = None
    if backend == "nccl":
        # Pin GPU to be used to process local rank (one GPU per process)
        # Specific hvd
        torch.cuda.set_device(hvd.local_rank())
        device = "cuda"
    else:
        device = "cpu"

    # Data preparation
    dataset = RndDataset(nb_samples=config["nb_samples"])

    # Specific hvd
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"] / hvd.size()),
        num_workers=1,
        sampler=train_sampler,
    )

    # Model, criterion, optimizer setup
    model = wide_resnet50_2(num_classes=100).to(device)
    criterion = NLLLoss().to(device)
    optimizer = SGD(model.parameters(), lr=0.001)

    # Specific hvd
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    # Specific hvd
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Training loop log param
    log_interval = config["log_interval"]

    def _train_step(batch_idx, data, target):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # Add a softmax layer
        probabilities = torch.nn.functional.softmax(output, dim=0)

        loss_val = criterion(probabilities, target)
        loss_val.backward()
        optimizer.step()

        if (batch_idx + 1) % (log_interval) == 0:
            print(
                "Process {}/{} Train Epoch: {} [{}/{}]\tLoss: {}".format(
                    hvd.local_rank(),
                    hvd.size(),
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_sampler),
                    loss_val.item(),
                )
            )

        return loss_val

    # Running _train_step for n_epochs
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            _train_step(batch_idx, data, target)

    # Specific hvd
    hvd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - Horovod")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--nb_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args_parsed = parser.parse_args()

    config = {
        "log_interval": args_parsed.log_interval,
        "batch_size": args_parsed.batch_size,
        "nb_samples": args_parsed.nb_samples,
    }

    args = (args_parsed.nproc_per_node, args_parsed.backend, config)

    # Specific hvd
    run(_mp_train, args=args, use_gloo=True, np=args_parsed.nproc_per_node)
