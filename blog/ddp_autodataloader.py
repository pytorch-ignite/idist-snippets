# Specific torch.distributed
train_sampler = DistributedSampler(dataset)

train_loader = DataLoader(
    dataset,
    batch_size=int(config["batch_size"] / world_size),
    num_workers=1,
    sampler=train_sampler,
)