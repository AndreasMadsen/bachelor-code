
import sys
import collections.abc
import numpy as np
import progressbar

from neural.learn.minibatch import minibatch

class GridSearch:
    def __init__(self, model, param_grid, seed=42, test_size=128, on_result=None, verbose=False):
        """Optimize hyperparameters using a grid search.

        Parameters
        ----------
        model: A neural model
            Should have a train, test and predict method

        param_grid : dict
            A dict instance describing the grid space.

        seed : integer
            The random seed used before reinitializing the neural network

        on_result: function (default None)
            Executed every time a search trail have been completed

        verbose : boolean
            If true progressiv information will be printed.
        """

        self._model = model
        self._param_grid = param_grid
        self._seed = seed
        self._test_size = test_size
        self._on_result = on_result
        self._verbose = verbose

        if (self._verbose): print("Initialized new GridSearch")

    def _search(self, trial):
        """This evil looking recursion finds all possibol combinations given
        the param_grid.
        """
        keys = sorted(self._param_grid.keys())

        def recursion(params, keys):
            if (len(keys) == 0):
                return trial(params)

            key = keys.pop()
            values = self._param_grid[key]

            if (isinstance(values, collections.abc.Iterable)):
                for val in values:
                    params[key] = val
                    recursion(params, list(keys))
            else:
                params[key] = values
                recursion(params, list(keys))

        recursion({}, keys)

    def _reset_and_optimize(self, train, params):
        # Save RNG state
        old_rng_state = np.random.get_state()

        # Reset model weights and optimization state
        np.random.seed(self._seed)
        self._model.reset_weights()
        self._model.reset_state()

        if (self._verbose and sys.stdout.isatty()):
            pbar = progressbar.ProgressBar(
                widgets=[
                    'Training: ', progressbar.Bar(),
                    progressbar.Percentage(), ' | ', progressbar.ETA()
                ],
                maxval=params['epochs']
            ).start()

            def on_mini_batch(model, index, epoch, loss):
                pbar.update(epoch)

            minibatch(self._model, train, on_mini_batch=on_mini_batch, **params)
            pbar.finish()

        elif (self._verbose):
            def on_epoch(model, index, epoch, loss):
                print('\tepoch: %d' % epoch_i)

            minibatch(self._model, train, on_epoch=on_epoch, **params)

        else:
            minibatch(self._model, train, **params)

        # Restore RNG state
        np.random.set_state(old_rng_state)

    def run(self, data):
        """Returns the test and training from the final iteration in each
        parameter combination.

        Parameters
        ----------
        data : Dataset
            A dataset instance

        Returns
        -------
        result : list, [combinations]
            A list of objects, each object contains `test_loss`, `train_loss`,
            `parameters` properties
        """
        test_dataset = data.range(0, self._test_size)
        train_sample = data.range(self._test_size, 2 * self._test_size)
        train_dataset = data.range(self._test_size, None)

        results = []

        def trial(params):
            if (self._verbose):
                print('\nRunning model: ')
                for key, val in params.items():
                    print('\t%s: %f' % (key, val))

            self._reset_and_optimize(train_dataset, params)

            result = {
                "test_loss": float(self._model.test(*test_dataset.astuple())),
                "train_loss": float(self._model.test(*train_sample.astuple())),
                "params": params
            }

            if (self._verbose):
                print('\tDone: train loss: %f, test loss: %f' % (
                    result['train_loss'], result['test_loss']
                ))

            if (self._on_result): self._on_result(result)
            results.append(result)
        self._search(trial)

        return results
