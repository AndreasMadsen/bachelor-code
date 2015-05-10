
def batch(model, train_dataset, epochs=100, on_epoch=None, **kwargs):
    if (on_epoch is not None): on_epoch(model, 0)
    for epoch_i in range(1, epochs + 1):
        model.train(*train_dataset.astuple(), **kwargs)

        if (on_epoch is not None): on_epoch(model, epoch_i)
