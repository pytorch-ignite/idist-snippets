import argparse

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from torch.nn import NLLLoss
from torch.optim import SGD
from torch.utils.data import Dataset
from torchvision.models import wide_resnet50_2

class RndDataset(Dataset):
    def __init__(self, nb_samples=128, labels=100):
        self._labels = labels
        self._nb_samples = nb_samples
        torch.randn(idist.get_rank())

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y


def _mp_train(rank, config):

    # Specific ignite.distributed
    print(idist.get_rank(), ': run with config:', config, '- backend=', idist.backend(), '- world size',
          idist.get_world_size())
    device = idist.device()

    # Data preparation:
    dataset = RndDataset(nb_samples=config['nb_samples'])

    # Specific ignite.distributed
    train_loader = idist.auto_dataloader(
        dataset, batch_size=config['batch_size']
    )

    # Model, criterion, optimizer setup
    model = idist.auto_model(wide_resnet50_2(num_classes=100))
    criterion = NLLLoss()
    optimizer = idist.auto_optim(SGD(model.parameters(), lr=0.01))

    # Training loop log param
    log_interval = config['log_interval']

    def _train_step(engine, batch):

        data = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        output = model(data)
        # Add a softmax layer
        probabilities = torch.nn.functional.softmax(output, dim=0)

        loss_val = criterion(probabilities, target)
        loss_val.backward()
        optimizer.step()

        return loss_val

    # Running the _train_step function on whole batch_data iterable only once
    trainer = Engine(_train_step)

    # Add a logger
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training():
        print('Process {}/{} Train Epoch: {} [{}/{}]\tLoss: {}'.format(idist.get_rank(), idist.get_world_size(),
                                                                       trainer.state.epoch,
                                                                       trainer.state.iteration * len(
                                                                           trainer.state.batch[0]),
                                                                       len(dataset) / idist.get_world_size(),
                                                                       trainer.state.output))

    trainer.run(train_loader, max_epochs=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pytorch Ignite - idist")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--nb_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args_parsed = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributed.launch, horovodrun, slurm)
    config = {'log_interval': args_parsed.log_interval,
              'batch_size': args_parsed.batch_size,
              'nb_samples': args_parsed.nb_samples}

    spawn_kwargs = dict()
    if args_parsed.nproc_per_node is not None:
        spawn_kwargs['nproc_per_node'] = args_parsed.nproc_per_node

    # Specific ignite.distributed
    with idist.Parallel(backend=args_parsed.backend, **spawn_kwargs) as parallel:
        parallel.run(_mp_train, config)
