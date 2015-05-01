
def batch(model, train_dataset, epochs=100, on_epoch=None, **kwargs):
    for epoch_i in range(0, epochs):
        model.train(*train_dataset, **kwargs)

        if (on_epoch is not None): on_epoch(model, epoch_i)
