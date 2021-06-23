model = resnet50(num_classes=100).cuda()

# Broadcast parameters from rank 0 to all other processes
hvd.broadcast_parameters(model.state_dict(), root_rank=0)