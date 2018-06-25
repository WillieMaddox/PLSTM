from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import LSTMStateTuple, LayerRNNCell


def random_exp_initializer(minval=0, maxval=None, seed=None, dtype=dtypes.float32):
    """Returns an initializer that generates tensors with an exponential distribution.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type.

    Returns:
      An initializer that generates tensors with an exponential distribution.
    """
    
    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.exp(random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed))
    
    return _initializer


class PhasedLSTMCell(LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
  
    The default non-peephole implementation is based on:
  
      http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
  
    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  
    The peephole implementation is based on:
  
      https://research.google.com/pubs/archive/43905.pdf
  
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.
  
    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """
    
    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 alpha=0.001,
                 r_on_init=0.05,
                 tau_init=6.,
                 manual_set=False,
                 trainable=True,
                 reuse=None):

        super(PhasedLSTMCell, self).__init__(_reuse=reuse)
        """Initialize the parameters for an LSTM cell.
    
        Args:
          num_units: int, The number of units in the LSTM cell
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)
        
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self.alpha = alpha
        self.r_on_init = r_on_init
        self.tau_init = tau_init
        
        self.manual_set = manual_set
        self.trainable = trainable
        
        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj) if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units) if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)

        self._kernel = self.add_variable(
            "kernel",
            shape=[h_depth + input_depth - 1, 4 * self._num_units],
            initializer=self._initializer,
            partitioner=maybe_partitioner)

        self._bias = self.add_variable(
            "bias",
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units], initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units], initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units], initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)

            self._proj_kernel = self.add_variable(
                "projection/kernel",
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size
    
    def call(self, inputs, state):
        """Run one step of LSTM.
    
        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "lstm_cell".
    
        Returns:
          A tuple containing:
    
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
    
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        
        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
        
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # --------------------------------------- #
        # ------------- PHASED LSTM ------------- #
        # ---------------- BEGIN ---------------- #
        # --------------------------------------- #

        i_size = input_size.value - 1  # -1 to extract time
        times = array_ops.slice(inputs, [0, i_size], [-1, 1])
        filtered_inputs = array_ops.slice(inputs, [0, 0], [-1, i_size])

        tau = vs.get_variable(
            "T", shape=[self._num_units],
            initializer=random_exp_initializer(0, self.tau_init) if not self.manual_set else init_ops.constant_initializer(self.tau_init),
            trainable=self.trainable, dtype=dtype)

        r_on = vs.get_variable(
            "R", shape=[self._num_units],
            initializer=init_ops.constant_initializer(self.r_on_init),
            trainable=self.trainable, dtype=dtype)

        s = vs.get_variable(
            "S", shape=[self._num_units],
            initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()) if not self.manual_set else init_ops.constant_initializer(0.),
            trainable=self.trainable, dtype=dtype)

        tau_broadcast = tf.expand_dims(tau, axis=0)
        r_on_broadcast = tf.expand_dims(r_on, axis=0)
        s_broadcast = tf.expand_dims(s, axis=0)

        r_on_broadcast = tf.abs(r_on_broadcast)
        tau_broadcast = tf.abs(tau_broadcast)
        times = tf.tile(times, [1, self._num_units])

        # calculate kronos gate
        phi = tf.div(tf.mod(tf.mod(times - s_broadcast, tau_broadcast) + tau_broadcast, tau_broadcast), tau_broadcast)
        is_up = tf.less(phi, (r_on_broadcast * 0.5))
        is_down = tf.logical_and(tf.less(phi, r_on_broadcast), tf.logical_not(is_up))

        k = tf.where(is_up, phi / (r_on_broadcast * 0.5), tf.where(is_down, 2. - 2. * (phi / r_on_broadcast), self.alpha * phi))

        lstm_matrix = math_ops.matmul(array_ops.concat([filtered_inputs, m_prev], 1), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        # --------------------------------------- #
        # ------------- PHASED LSTM ------------- #
        # ----------------- END ----------------- #
        # --------------------------------------- #

        i, j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)

        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type

        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:

            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        # APPLY KRONOS GATE
        c = k * c + (1. - k) * c_prev
        m = k * m + (1. - k) * m_prev
        # END KRONOS GATE
        
        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else array_ops.concat([c, m], 1))
        return m, new_state
