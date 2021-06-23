optimizer = idist.auto_optim(SGD(model.parameters(), lr=0.01))

optimizer.step()