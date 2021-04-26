import argparse

import torch
import torch.distributed as dist
from torch.multiprocessing import start_processes
from torch.nn import NLLLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import Dataset
from torchvision.models import wide_resnet50_2


class RndDataset(Dataset):
    def __init__(self, nb_samples=128, labels=100):
        self._labels = labels
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y


def _mp_train(rank, world_size, backend, config):
    dist.init_process_group(backend, init_method="tcp://0.0.0.0:2233", world_size=world_size, rank=rank)

    device = None
    if backend == 'nccl':
        torch.cuda.set_device(rank)
        device = 'cuda'
    else:
        device = 'cpu'

    print(dist.get_rank(), ': run with config:', config, ' - backend=', backend)

    dataset = RndDataset(nb_samples=config['nb_samples'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size']/world_size,
        num_workers=max(4 / world_size, 1),
        sampler=train_sampler
    )

    # Model, criterion, optimizer setup
    model = wide_resnet50_2(num_classes=100).to(device)
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    if backend == 'nccl':
        model = DDP(model, device_ids=[rank])
    elif backend == 'gloo':
        model = DDP(model)

    log_interval = config['log_interval']

    def _train_step(batch_idx, data, target):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss_val = criterion(output, target)
        loss_val.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Process {}/{} Train Epoch: {} [{}/{}]\tLoss: {}'.format(dist.get_rank(), dist.get_world_size(),
                                                                           epoch, batch_idx * len(data),
                                                                           len(train_sampler), loss_val.item()))

    # Running _train_step for n_epochs
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            _train_step(batch_idx, data, target)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Torch Native - DDP")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--nb_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args_parsed = parser.parse_args()

    assert dist.is_available()
    if args_parsed.backend == "nccl":
        assert torch.cuda.is_available()
        assert dist.is_nccl_available()
    elif args_parsed.backend == "gloo":
        assert dist.is_gloo_available()
    else:
        raise ValueError(f"unvalid backend `{args_parsed.backend}` (valid: `gloo` or `nccl`)")

    config = {'log_interval': args_parsed.log_interval,
              'batch_size': args_parsed.batch_size,
              'nb_samples': args_parsed.nb_samples}

    args = (args_parsed.nproc_per_node, args_parsed.backend, config)

    start_processes(
        _mp_train, args=args, nprocs=args_parsed.nproc_per_node, start_method="spawn"
    )
