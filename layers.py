import keras
import keras.backend as K


class ExpandLayer(keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(ExpandLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        input_shape.insert(ax+1, 1)
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        return dict(axis=self.axis)


class SqueezeLayer(ExpandLayer):

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        if input_shape[ax] == 1:
            input_shape.pop(ax)
        else:
            raise ValueError('Dimension ', ax, 'is not equal to 1!')
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.squeeze(inputs, axis=self.axis)


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
            energy = K.concatenate((inputs, h_tm1))
        elif self.style is self.ATT_STYLES[1]:
            h_tm1 = K.repeat_elements(K.expand_dims(h_tm1), self.channels, -1)
            energy = K.concatenate((self.input_tensor_hack, h_tm1), 1)
            energy = K.permute_dimensions(energy, (0, 2, 1))
        else:
            raise NotImplementedError('{0}: not implemented'.format(self.style))

        for i, kernel in enumerate(self.attention_kernels):
            energy = K.dot(energy, kernel)
            if self.use_bias:
                energy = K.bias_add(energy, self.attention_bias[i])
            energy = self.activation(energy)

        alpha = K.softmax(energy)

        if self.style is self.ATT_STYLES[0]:
            inputs = inputs * alpha
        elif self.style is self.ATT_STYLES[1]:
            alpha = K.permute_dimensions(alpha, (0, 2, 1))
            weighted = self.input_tensor_hack * alpha
            inputs = K.sum(weighted, 1)

        return super(AttentionLSTMIn, self).step(inputs, states)