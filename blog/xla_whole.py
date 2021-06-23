def training(rank, world_size, backend, config):
    # Specific xla
    device = xm.xla_device()

    # Data preparation
    dataset = ...

    # Specific xla
    train_sampler = DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"] / xm.xrt_world_size()),
        num_workers=1,
        sampler=train_sampler,
    )

    # Specific xla
    para_loader = pl.MpDeviceLoader(train_loader, device)

    # Model, criterion, optimizer setup
    model = resnet50(num_classes=100).to(device)
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    ...

    def train_step(batch_idx, data, target):

        data = data
        target = target
        ...
        output = model(data)
        ...
        loss_val = ...
        
        xm.optimizer_step(optimizer)
        
        return loss_val

    # Running train_step for n_epochs
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(para_loader):
            train_step(batch_idx, data, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - XLA")
    parser.add_argument("--backend", type=str, default="xla-tpu")
    parser.add_argument("--nproc_per_node", type=int, default=8)

    ...
    
    args = (args_parsed.nproc_per_node, args_parsed.backend, config)
    # Specific xla
    xmp.spawn(training, args=args, nprocs=args_parsed.nproc_per_node)