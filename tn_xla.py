import argparse

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.nn import NLLLoss
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
    print(xm.get_ordinal(), ': run with config:', config, '- backend=', backend)

    device = xm.xla_device()

    dataset = RndDataset(nb_samples=config['nb_samples'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=xm.xrt_world_size(),
                                                                    rank=xm.get_ordinal(),
                                                                    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=max(4 / xm.xrt_world_size(), 1),
        sampler=train_sampler
    )

    para_loader = pl.ParallelLoader(train_loader, [device])

    # Model, criterion, optimizer setup
    model = wide_resnet50_2(num_classes=100).to(device)
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    log_interval = config['log_interval']

    def _train_step(batch_idx, data, target):

        data = data
        target = target

        optimizer.zero_grad()
        output = model(data)

        loss_val = criterion(output, target)
        loss_val.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Process {}/{} Train Epoch: {} [{}/{}]\tLoss: {}'.format(xm.get_ordinal(), xm.xrt_world_size(),
                                                                           epoch, batch_idx * len(data),
                                                                           len(train_sampler), loss_val.item()))

    # Running _train_step for n_epochs
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(para_loader):
            _train_step(batch_idx, data, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Torch Native - XLA")
    parser.add_argument("--backend", type=str, default="xla-tpu")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--nb_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args_parsed = parser.parse_args()

    assert args_parsed.backend == "xla-tpu"

    config = {'log_interval': args_parsed.log_interval,
              'batch_size': args_parsed.batch_size,
              'nb_samples': args_parsed.nb_samples}

    args = (args_parsed.nproc_per_node, args_parsed.backend, config)
    xmp.spawn(_mp_train, args=args, nprocs=args_parsed.nproc_per_node)
