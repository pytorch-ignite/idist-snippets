# Specific xla
device = xm.xla_device()

# Model, criterion, optimizer setup
model = resnet50(num_classes=100).to(device)