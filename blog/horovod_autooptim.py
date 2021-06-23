optimizer = SGD(model.parameters(), lr=0.001)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters()
)

optimizer.step()
