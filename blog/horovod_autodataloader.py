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