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