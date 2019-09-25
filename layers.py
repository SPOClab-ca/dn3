import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow as tf


class ExpandLayer(keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(ExpandLayer, self).__init__(**kwargs)

    def compute_output_signature(self, input_signature):
        ax = self.axis
        input_signature = list(input_signature)
        if ax < 0:
            ax = len(input_signature) + ax
        input_signature.insert(ax + 1, 1)
        return tuple(input_signature)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        return dict(axis=self.axis)


class SqueezeLayer(ExpandLayer):

    def compute_output_signature(self, input_signature):
        ax = self.axis
        input_signature = list(input_signature)
        if ax < 0:
            ax = len(input_signature) + ax
        if input_signature[ax] == 1:
            input_signature.pop(ax)
        else:
            raise ValueError('Dimension ', ax, 'is not equal to 1!')
        return tuple(input_signature)

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=self.axis)


# need these for ShallowConvNet
def square(x):
    return tf.math.square(x)


def log(x):
    return tf.math.log(tf.keras.backend.clip(x, min_value=1e-7, max_value=10000))


class AttentionLSTMIn(keras.layers.LSTM):
    """
    Keras LSTM layer (all keyword arguments preserved) with the addition of attention weights
    Attention weights are calculated as a function of the previous hidden state to the current LSTM step.
    Weights are applied either locally (across channels at current timestep) or globally (weight each sequence element
    of each channel).
    """
    ATT_STYLES = ['local', 'global']

    def __init__(self, units, alignment_depth: int = 1, style='local', alignment_units=None, implementation=2,
                 **kwargs):
        implementation = implementation if implementation > 0 else 2
        alignment_depth = max(0, alignment_depth)
        if isinstance(alignment_units, (list, tuple)):
            self.alignment_units = [int(x) for x in alignment_units]
            self.alignment_depth = len(self.alignment_units)
        else:
            self.alignment_depth = alignment_depth
            self.alignment_units = [alignment_units if alignment_units else units for _ in range(alignment_depth)]
        if style not in self.ATT_STYLES:
            raise TypeError('Could not understand style: ' + style)
        else:
            self.style = style
        super(AttentionLSTMIn, self).__init__(units, implementation=implementation, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) > 2
        self.samples = input_shape[1]
        self.channels = input_shape[2]

        if self.style is self.ATT_STYLES[0]:
            # local attends over input vector
            units = [self.units + input_shape[-1]] + self.alignment_units + [self.channels]
        else:
            # global attends over the whole sequence for each feature
            units = [self.units + input_shape[1]] + self.alignment_units + [self.samples]
        self.attention_kernels = [self.add_weight(shape=(units[i-1], units[i]),
                                                name='attention_kernel_{0}'.format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True,
                                                constraint=self.kernel_constraint)
                                  for i in range(1, len(units))]

        if self.use_bias:
            self.attention_bias = [self.add_weight(shape=(u,),
                                                   name='attention_bias',
                                                   trainable=True,
                                                   initializer=self.bias_initializer,
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint)
                                   for u in units[1:]]
        else:
            self.attention_bias = None
        super(AttentionLSTMIn, self).build(input_shape)

    def preprocess_input(self, inputs, training=None):
        self.input_tensor_hack = inputs
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]

        if self.style is self.ATT_STYLES[0]:
            energy = tf.concatenate((inputs, h_tm1))
        elif self.style is self.ATT_STYLES[1]:
            h_tm1 = tf.repeat_elements(tf.expand_dims(h_tm1), self.channels, -1)
            energy = tf.concatenate((self.input_tensor_hack, h_tm1), 1)
            energy = tf.permute_dimensions(energy, (0, 2, 1))
        else:
            raise NotImplementedError('{0}: not implemented'.format(self.style))

        for i, kernel in enumerate(self.attention_kernels):
            energy = tf.dot(energy, kernel)
            if self.use_bias:
                energy = tf.bias_add(energy, self.attention_bias[i])
            energy = self.activation(energy)

        alpha = tf.softmax(energy)

        if self.style is self.ATT_STYLES[0]:
            inputs = inputs * alpha
        elif self.style is self.ATT_STYLES[1]:
            alpha = tf.permute_dimensions(alpha, (0, 2, 1))
            weighted = self.input_tensor_hack * alpha
            inputs = tf.sum(weighted, 1)

        return super(AttentionLSTMIn, self).step(inputs, states)


def scnn_spatial_filter(x, filters, depth, do_rate=0, activation='relu', batch_norm=True,
                        residual='dense', data_format='channels_first'):
    """
    Creates a deep convolutional spatial filter without temporal aspect
    Parameters
    ----------
    x: The input tensor, of shape (batch, channels, temporal)
    filters: The number of filters per layer, if residual is set to 'dense', then this is the growth rate.
    depth: The number of layers needed to sweep the input channels
    do_rate: Dropout rate per layer
    activation: Activation at each layer
    batch_norm: Whether to include batch normalization at each step
    residual: One of ['layerwise', 'netwise', 'dense'] or None. Layerwise adds a residual connection from the input
              (through 1x1 convolutions) to the end of each layer. Dense is like densenet, concatenating previous
              filters at each step. Netwise adds one residual from start to end. None ignores.

    Returns
    -------
    Output tensor of shape (batch, filters, temporal)
    """
    channels = x.shape[1]
    kernels = [(channels // depth, 1) for _ in range(depth - 1)]
    kernels += [(channels - sum(x[1] for x in kernels) + depth - 1, 1)]
    inp = x
    for i in range(depth):
        l_begin = x
        x = ExpandLayer(axis=1)(x)
        x = Conv2D(filters, (channels + filters*i, 1), activation=activation, padding='valid',
                   data_format=data_format)(x)
        x = SqueezeLayer(axis=2)(x)
        x = SpatialDropout1D(do_rate)(x)
        if batch_norm:
            x = BatchNormalization(axis=1 if data_format == 'channels_first' else -1)(x)
        if not isinstance(residual, str):
            pass
        elif residual.lower() == 'dense':
            x = Concatenate(axis=1)([x, l_begin])
        elif residual.lower() == 'layerwise':
            x = Add()([x, Conv1D(filters, 1, data_format=data_format)(inp)])
    if residual is not None and residual.lower() == 'netwise':
        x = Add()([x, Conv1D(filters, 1, data_format=data_format)(inp)])
    return x


def scnn_temporal_filter(x, growth, depth, t_len, dropout=0., activation='relu', data_format='channels_first'):
    x = ExpandLayer(axis=1)(x)
    for i in range(depth):
        l_start = x
        x = Conv2D(growth, (1, t_len), dilation_rate=(1, depth-i), padding='same', data_format=data_format,
                   activation=activation)(x)
        x = SpatialDropout2D(dropout)(x)
        x = BatchNormalization(axis=1 if data_format == 'channels_first' else -1)(x)
        x = Concatenate(axis=1)([x, l_start])
    return x


# The dense layers mostly a simple modification of the torchvision
def dense_layer_1d(in_tensor, growth_rate, bn_size, drop_rate, data_format='channels_first', activation=ReLU):
       in_tensor = BatchNormalization(axis=1 if data_format == 'channels_first' else -1)(in_tensor)
       in_tensor = activation()(in_tensor)
       in_tensor = Conv1D(bn_size * growth_rate, kernel_size=1, strides=1, use_bias=False, data_format=data_format,
                          padding='same')(in_tensor)

       in_tensor = BatchNormalization(axis=1 if data_format == 'channels_first' else -1)(in_tensor)
       in_tensor = activation()(in_tensor)
       in_tensor = Conv1D(growth_rate, kernel_size=3, strides=1, use_bias=False, data_format=data_format,
                          padding='same')(in_tensor)
       in_tensor = SpatialDropout1D(rate=drop_rate)(in_tensor)

       return in_tensor


def dense_block_1d(in_tensor, num_layers, bn_size, growth_rate, drop_rate=0.5,
                   data_format='channels_first', activation=ReLU):
    for i in range(num_layers):
        out = dense_layer_1d(in_tensor, growth_rate, bn_size, drop_rate, data_format=data_format, activation=activation)
        in_tensor = Concatenate(axis=1 if data_format == 'channels_first' else -1)([in_tensor, out])
    return in_tensor


def transition(in_tensor, num_output_features, pool=2, activation=ReLU, data_format='channels_first'):
    in_tensor = BatchNormalization(axis=1 if data_format == 'channels_first' else -1)(in_tensor)
    in_tensor = activation()(in_tensor)
    in_tensor = Conv1D(num_output_features, kernel_size=1, strides=1, use_bias=False, data_format=data_format,
                       padding='same')(in_tensor)
    in_tensor = AveragePooling1D(pool, strides=pool)(in_tensor)
    return in_tensor
