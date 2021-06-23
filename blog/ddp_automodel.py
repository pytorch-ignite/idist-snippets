model = resnet50(num_classes=100).cuda()

# Specific torch.distributed
model = DDP(model, device_ids=[rank])