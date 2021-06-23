def training(rank, config):

    # Specific ignite.distributed
    device = idist.device()

    # Data preparation:
    dataset = ...

    # Specific ignite.distributed
    train_loader = idist.auto_dataloader(dataset, batch_size=config["batch_size"])

    # Model, criterion, optimizer setup
    model = idist.auto_model(resnet50(num_classes=100))
    criterion = NLLLoss()
    optimizer = idist.auto_optim(SGD(model.parameters(), lr=0.01))

    ...
    
    def train_step(engine, batch):

        data = batch[0].to(device)
        target = batch[1].to(device)
        ...
        output = model(data)
        ...
        loss_val = ...
        
        return loss_val

    # Runs train_step on whole batch_data iterable only once
    trainer = Engine(train_step)

    # Specific Pytorch-Ignite
    trainer.run(train_loader, max_epochs=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytorch Ignite - idist")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)

    ...
    # Specific ignite.distributed
    with idist.Parallel(backend=args_parsed.backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)