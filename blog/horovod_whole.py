def training(world_size, backend, config):
    # Specific hvd
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    # Specific hvd
    torch.cuda.set_device(hvd.local_rank())
    
    # Data preparation
    dataset = ...

    # Specific hvd
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"] / hvd.size()),
        num_workers=1,
        sampler=train_sampler,
    )

    # Model, criterion, optimizer setup
    model = resnet50(num_classes=100)
    model.cuda()
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    # Specific hvd
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    # Specific hvd
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    ...

    def train_step(batch_idx, data, target):

        data, target = data.cuda(), target.cuda()
        ...
        output = model(data)
        ...
        loss_val = ...

        return loss_val

    # Running train_step for n_epochs
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            train_step(batch_idx, data, target)

    # Specific hvd
    hvd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - Horovod")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--nproc_per_node", type=int, default=2)

    ...

    args = (args_parsed.nproc_per_node, args_parsed.backend, config)

    # Specific hvd
    run(training, args=args, use_gloo=True, np=args_parsed.nproc_per_node)