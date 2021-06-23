optimizer = SGD(model.parameters(), lr=0.01)

xm.optimizer_step(optimizer)