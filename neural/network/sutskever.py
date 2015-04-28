
import itertools

import numpy as np
import theano
import theano.tensor as T

from neural.layer.lstm import LSTM
from neural.layer.rnn import RNN

from neural.network._base import BaseAbstraction
from neural.network._optimizer import OptimizerAbstraction


class SutskeverNetwork(OptimizerAbstraction):
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, max_output_size=100, **kwargs):
        OptimizerAbstraction.__init__(self, **kwargs)

        self._input = T.tensor3('x')
        self._target = T.imatrix('t')

        # encoder -> decoder
        self._encoder = Encoder(self._input)
        self._decoder = Decoder(self._input, maxlength=max_output_size)

    def test_value(self, x, t):
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def set_encoder_input(self, layer):
        self._encoder.set_input(layer)

    def push_encoder_layer(self, layer):
        self._encoder.push_layer(layer)

    def set_decoder_input(self, layer):
        self._decoder.set_input(layer)

    def push_decoder_layer(self, layer):
        self._decoder.push_layer(layer)
        self._output_layer = layer

    def weight_list(self):
        """
        Create a list containing all the network weights
        """
        return self._encoder.weight_list() + self._decoder.weight_list()

    def reset_weights(self):
        """
        Will reinitialize all weights using a random number generator
        """
        if (self._verbose): print('Weights reset')
        self._decoder.reset_weights()
        self._encoder.reset_weights()

    def forward_pass(self, x):
        (s_enc, b_enc) = self._encoder.forward_pass(x)
        (eois, log_y, y) = self._decoder.forward_pass(s_enc, b_enc)

        return (eois, log_y, y)

    def _preloss_scanner(self, y_i, eosi_i, t_i, y_pad, dims, time):
        # Get length of y seqence including the first <EOS>
        yend = eosi_i + 1

        # Since t is already padded with <EOS>, we can just copy the
        # entire sequence.
        tend = t_i.shape[0]

        # Create a new y sequence with T elements
        y2_i = T.zeros((dims, time), dtype='float32')
        # Keep the actual y elements
        y2_i = T.set_subtensor(y2_i[:, :yend], y_i[:, :yend])
        # Add ignore padding to y2 for the remaining elments
        y2_i = T.set_subtensor(y2_i[:, yend:], y_pad)

        # Createa a new t seqnece with T elements
        t2_i = T.zeros((time, ), dtype='int32')
        # Keep the actual t elements
        t2_i = T.set_subtensor(t2_i[:tend], t_i)
        # Add <EOS> padding to t2 for the remaining elments,
        # the y2 padding will ignore this
        # -- No code, since 0 from T.zeros is <EOS>

        return [y2_i, t2_i]

    def _preloss(self, eosi, log_y, y, t):
        # Create an <EOS> collum vector
        y_pad = T.ones((y.shape[1], ), dtype='float32')
        y_pad = y_pad * T.log(0.01 / T.cast(y.shape[1], 'float32'))
        y_pad = T.set_subtensor(y_pad[0], T.log(0.99))
        y_pad = y_pad.dimshuffle((0, 'x'))

        (log_y_pad, t_pad), _ = theano.scan(
            fn=self._preloss_scanner,
            sequences=[log_y, eosi, t],
            outputs_info=[None, None],
            non_sequences=[
                y_pad,
                y.shape[1],
                T.max([y.shape[2], t.shape[1]])
            ],
            name='sutskever_loss'
        )

        return (log_y_pad, t_pad)

    def compile(self):
        # Sutskever is numerically unstable unless stable prelog is used
        assert(self._output_layer._add_log)

        # The input decoder much match its softmax output
        assert(self._decoder._layers[+0].output_size == self._decoder._layers[-1].output_size)
        # The hidden encoder output much match the hidden decoder intialization
        assert(self._encoder._layers[-1].output_size == self._decoder._layers[+1].output_size)

        super().compile()

class Encoder(BaseAbstraction):
    """
    The encoder is like a normal forward scanner but doesn't have an output
    layer applied to it. It also only outputs the last time iteration. This
    is how the Sutskever network is implemented. Better aproches may exists.

    This implementation also uses a vector, indicate when the sequences starts
    relative to the other sequences in the minibatch.
    """
    def __init__(self, x_input, **kwargs):
        BaseAbstraction.__init__(self, **kwargs)
        # self._input is only used by the layers, to infer the batch size.
        self._input = x_input

    def _forward_scanner(self, x_t, *args):
        """
        Defines the forward equations for each time step.
        """
        args = list(args)

        # Intialize loop
        all_outputs = []
        curr = 0
        prev_output = x_t

        # Mask the inputs
        mask = T.eq(x_t[:, 0], 1).dimshuffle((0, 'x'))  # is <EOF>

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = layer.infer_taps()
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps], mask=mask)

            curr += taps
            all_outputs += layer_outputs
            # the last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        return all_outputs

    def forward_pass(self, x):
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
                x.transpose(2, 0, 1)  # iterate (time), row (observations), col (dims)
            ],
            outputs_info=self._outputs_info_list(),
            name='sutskever_encoder'
        )
        b_enc = self._last_output(outputs)

        # the last output is assumed to be the network output, take the
        # last time iteration. Return value shape is: row (observations), col (dims)
        if (isinstance(self._layers[-1], RNN)):
            return (None, b_enc[-1, :, :])
        else:
            s_enc = outputs[-2]
            return (s_enc[-1, :, :], b_enc[-1, :, :])

class SutskeverEncoder(Encoder, OptimizerAbstraction):
    def __init__(self, **kwargs):
        self._input = T.tensor3('x')
        self._target = T.ivector('t')

        Encoder.__init__(self, self._input, **kwargs)
        OptimizerAbstraction.__init__(self, **kwargs)

    def _preloss(self, y_log, y, t):
        return (y_log, t)

    def test_value(self, x, b_enc):
        self._input.tag.test_value = x
        self._target.tag.test_value = b_enc


class Decoder(BaseAbstraction):
    """
    The decoder takes the encoder output for the last time iteration and
    passes it intro a forward iteration. The next output iteration is then
    obtained by letting the hidden output of $t$ go into $t + 1$. The <EOF>
    tag signals when to stop.
    """
    def __init__(self, x_input, maxlength=100, **kwargs):
        BaseAbstraction.__init__(self, **kwargs)
        # self._input is only used by the layers, to infer the batch size.
        self._input = x_input
        self._maxlength = maxlength

    def _forward_scanner(self, t, eosi, mask, *args):
        """
        Defines the forward equations for each time step.
        """
        # The last argument is the network output (after softmax)
        args = list(args)
        y = args.pop()

        # Intialize layer loop
        all_outputs = []
        curr = 0
        prev_output = y
        mask_col = mask.dimshuffle((0, 'x'))

        # Loop though each layer and apply send the current layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = layer.infer_taps()
            # It can be assumed that the last value in `layer_outputs` is the
            # actual layer output. The tuple may contain other values, such
            # as the current cell state. They won't be send to the next layer.
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps], mask=mask_col)
            curr += taps

            # Concatenate all outputs
            all_outputs += layer_outputs
            # The last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        # Update the mask, if <EOS> was returned by the last iteration.
        # At this point the prev_output is the last layer_output and is
        # thus the network output.
        new_mask = T.eq(T.argmax(prev_output, axis=1), 0)
        # Update eosi where new observations have ended `new_mask - mask`
        eosi = T.set_subtensor(eosi[T.nonzero(new_mask - mask)[0]], t)

        all_outputs = [eosi, new_mask] + all_outputs

        # Stop when all sequences are masked
        return (all_outputs, theano.scan_module.until(T.all(new_mask)))

    def _outputs_info_list(self, s_enc, b_enc):
        outputs_info = super()._outputs_info_list()

        # 1) Replace the initial b_{t_0} with b_enc for the first layer
        if (isinstance(self._layers[1], LSTM)):
            outputs_info[1] = b_enc
            if (s_enc is not None):
                outputs_info[0] = s_enc

        elif (isinstance(self._layers[1], RNN)):
            outputs_info[0] = b_enc
            if (s_enc is not None):
                raise TypeError('s_enc could not be transferred to RNN layer')

        else:
            raise NotImplemented

        # 2) Replace the initial y_{t_0} with 0, and add taps = -1
        y = T.zeros((b_enc.shape[0], self._layers[-1].output_size))
        y.name = 'y'
        outputs_info[-1] = y

        # 3) Initialize with no mask
        mask = T.zeros((b_enc.shape[0], ), dtype='int8')
        mask.name = 'mask'
        outputs_info = [mask] + outputs_info

        # 4) Initialize the <EOS> index counter, to be the last possibol
        # index. This way, if the scanner doesn't end by <EOS> the eosi
        # will still be meaningful in the used context.
        eosi = (self._maxlength - 1) * T.ones((b_enc.shape[0], ), dtype='int32')
        eosi.name = 'eosi'
        outputs_info = [eosi] + outputs_info

        # outputs_info will look like [eois, mask, s_enc, b_enc, ..., y]
        return outputs_info

    def forward_pass(self, s_enc, b_enc):
        """
        Setup equations for the forward pass
        """
        # To create an <EOS> index vector (eosi) the time index is needed.
        # Use `arange` to generate a vector with all the time indexes. The
        # `scan` may finish before because the `until` condition becomes true.
        time_seq = T.arange(0, self._maxlength)
        time_seq.name = 'time'

        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=[time_seq],
            outputs_info=self._outputs_info_list(s_enc, b_enc),
            name='sutskever_decoder'
        )

        eosi = outputs[0][-1, :]

        # The scan output have the shape (time, dims, observations)
        log_y = outputs[-2].transpose(1, 2, 0)
        y = outputs[-1].transpose(1, 2, 0)

        return (eosi, log_y, y)


class SutskeverDecoder(Decoder, OptimizerAbstraction):
    def __init__(self, **kwargs):
        self._input = T.matrix('b_enc')
        self._target = T.imatrix('t')

        Decoder.__init__(self, self._input, **kwargs)
        OptimizerAbstraction.__init__(self, **kwargs)

    def test_value(self, x, t):
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def forward_pass(self, b_enc):
        # Use the hidden input as the cell input too if it is a LSTM layer
        s_enc = b_enc if isinstance(self._layers[1], LSTM) else None
        return Decoder.forward_pass(self, s_enc, b_enc)

    def _preloss_scanner(self, *args, **kwargs):
        return SutskeverNetwork._preloss_scanner(self, *args, **kwargs)

    def _preloss(self, *args, **kwargs):
        return SutskeverNetwork._preloss(self, *args, **kwargs)
