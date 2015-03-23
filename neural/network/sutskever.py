
import itertools

import numpy as np
import theano
import theano.tensor as T

from neural.network._base import BaseAbstraction
from neural.network._optimizer import OptimizerAbstraction

class SutskeverNetwork(OptimizerAbstraction):
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, **kwargs):
        OptimizerAbstraction.__init__(self, **kwargs)

        self._input = T.tensor3('x')
        self._target = T.imatrix('t')

        # encoder -> decoder
        self._encoder = Encoder(self._input)
        self._decoder = Decoder()

    def weight_list(self):
        """
        Create a list containing all the network weights
        """
        return self._encoder.weight_list() + self._decoder.weight_list()

    def test_value(self, x, t):
        # TODO: insert <EOF> tag
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def forward_pass(self, x):
        # TODO: insert <EOF> tag
        # TODO: build mask vector (start_mask)
        b_enc_end = self._encoder.forward_pass(x, start_mask)
        (end_mask, y) = self._decoder.forward_pass(b_enc_end)

        return (end_mask, y)

class Encoder(BaseAbstraction):
    """
    The encoder is like a normal forward scanner but doesn't have an output
    layer applied to it. It also only outputs the last time iteration. This
    is how the Sutskever network is implemented. Better aproches may exists.

    This implementation also uses a vector, indicate when the sequences starts
    relative to the other sequences in the minibatch.
    """
    def __init__(self, x_input, **kwargs):
        super().__init__(**kwargs)
        self._input = x_input

    def _forward_scanner(self, t, x_t, *args):
        """
        Defines the forward equations for each time step.
        """
        args = list(args)
        start_mask = args.pop()

        # Intialize loop
        all_outputs = []
        curr = 0
        prev_output = x_t

        # Mask the inputs
        mask = (t >= start_mask).nonzero()[0]
        prev_output = prev_output[mask, :]
        args = [data[mask, :] for data in args]

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = self._infer_taps(layer)
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps])

            curr += taps
            all_outputs += layer_outputs
            # the last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        print(prev_output.tag.test_value)

        # Demask the outputs
        all_outputs_demask = []
        for output in all_outputs:
            output_demask = T.zeros((x_t.shape[0], output.shape[1]))
            print('demask:', output_demask.tag.test_value)
            print('with:', output.tag.test_value)

            # TODO: consider inplace optimization
            T.set_subtensor(output_demask[mask, :], output)
            print('demask:', output_demask.tag.test_value)

            all_outputs_demask.append(output_demask)

        return all_outputs_demask

    def forward_pass(self, x, start_mask):
        """
        Setup equations for the forward pass
        """
        # because scan assmes the iterable is the first tensor dimension, x is
        # transposed into (time, obs, dims).
        # When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
        # this is then transposed back to its original format
        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=[
                T.arange(0, x.shape[2]),  # iterate a time index
                x.transpose(2, 0, 1)  # iterate (time), row (observations), col (dims)
            ],
            outputs_info=self._outputs_info_list(),
            non_sequences=[start_mask]
        )
        b_enc = self._last_output(outputs)

        # the last output is assumed to be the network output, take the
        # last time iteration. Return value shape is: row (observations), col (dims)
        return b_enc[-1, :, :]

class Decoder(BaseAbstraction):
    """
    The decoder takes the encoder output for the last time iteration and
    passes it intro a forward iteration. The next output iteration is then
    obtained by letting the hidden output of $t$ go into $t + 1$. The <EOF>
    tag signals when to stop.
    """
    def __init__(self, maxlength=100, **kwargs):
        super().__init__(**kwargs)
        self._maxlength = maxlength

    def _forward_scanner(self, t, end_mask, b_enc, *args):
        """
        Defines the forward equations for each time step.
        """
        # Intialize loop
        all_outputs = []
        curr = 0
        prev_output = b_enc

        # Mask the inputs
        mask = end_mask == -1
        prev_output = prev_output[mask, :]
        args = [data[mask, :] for data in args]

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = self._infer_taps(layer)
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps])

            curr += taps
            all_outputs += layer_outputs
            # the last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        # Demask the outputs
        all_outputs_demask = []
        for output in all_outputs:
            # NOTE: this does not carry the result of previous iterations,
            # the solution is to use the mask array to merge results from
            # all time iterations.
            output_demask = T.zeros((x_t.shape[0], output.shape[1]))
            T.set_subtensor(output_demask[mask, :], output)
            all_outputs_demask.append(output_demask)

        # Select results
        b_dec = all_outputs_demask[-2]
        y_dec = all_outputs_demask[-1]

        # Update mask array
        eos_mask = T.argmax(y_dec, axis=1) == 0
        T.set_subtensor(eos_mask[end_mask >= 0], 0)  # Set false for skiped obs
        T.set_subtensor(end_mask[eos_mask], t)  # Set t for done sequences

        return (
            [end_mask, b_dec] + all_outputs_demask,
            theano.scan_module.until(T.all(mask))
        )

    def forward_pass(self, b_enc):
        """
        Setup equations for the forward pass
        """
        # because scan assmes the iterable is the first tensor dimension, x is
        # transposed into (time, obs, dims).
        # When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
        # this is then transposed back to its original format
        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=T.arange(0, self._maxlength),
            outputs_info=[
                -1 * T.ones(b_enc.shape[0], dtype='int32'),
                b_enc
            ] + self._outputs_info_list()
        )

        # Select the important arrays
        end_mask = (outputs[0])[-1, :]
        y = outputs[-1].transpose(1, 2, 0)

        return (end_mask, y)
