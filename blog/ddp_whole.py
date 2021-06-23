def training(rank, world_size, backend, config):

    # Specific torch.distributed
    dist.init_process_group(
        backend, init_method="tcp://0.0.0.0:2233", world_size=world_size, rank=rank
    )

    torch.cuda.set_device(rank)

    # Data preparation
    dataset = ...

    # Specific torch.distributed
    train_sampler = DistributedSampler(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"] / world_size),
        num_workers=1,
        sampler=train_sampler,
    )

    # Model, criterion, optimizer setup
    model = resnet50(num_classes=100).cuda()
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Specific torch.distributed
    model = DDP(model, device_ids=[rank])

    ...

    def train_step(batch_idx, data, target):

        data = data.cuda()
        target = target.cuda()
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

    # Specific torch.distributed
    dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - DDP")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    
    ...

    args = (args_parsed.nproc_per_node, args_parsed.backend, config)

    # Specific torch.distributed
    start_processes(
        training, args=args, nprocs=args_parsed.nproc_per_node, start_method="spawn"
    )
