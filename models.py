from .layers import *


def ShallowFBCSP(inputshape, outputshape):
    model = keras.models.Sequential()
    model.add(ExpandLayer(input_shape=inputshape))
    model.add(keras.layers.Conv2D(40, (1, 25), activation='linear', data_format='channels_last'))
    model.add(keras.layers.Conv2D(40, (25, 1), activation='linear', data_format='channels_last'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(K.square))
    model.add(keras.layers.AveragePooling2D((1, 75), 15))
    model.add(keras.layers.Activation(lambda x: K.log(K.maximum(x, K.constant(1e-6)))))
    model.add(keras.layers.Dropout(0.5))

    # Output convolution
    model.add(keras.layers.Conv2D(
        10, (1, 27), activation='linear', data_format='channels_last',
    ))
    model.add(keras.layers.BatchNormalization())

    # Classifier
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))
    return model


def BestSCNN(inputshape, outputshape, params=None):

    params = dict() if params is None else params

    temp_layers = int(params.get('temp_layers', 4))
    steps = int(params.get('steps', 2))
    temporal = int(params.get('temporal', 24))
    temp_pool = int(params.get('temp_pool', 20))
    lunits = [int(x) for x in params.get('lunits', [200, 40])]
    activation = params.get('activation', keras.activations.selu)
    reg = float(params.get('regularization', 0.01))
    do = min(1., max(0., float(params.get('dropout', 0.55))))

    convs = [inputshape[-1] // steps for _ in range(1, steps)]
    convs += [inputshape[-1] - sum(convs) + len(convs)]

    ins = keras.layers.Input(inputshape)

    conv = ExpandLayer()(ins)

    for i, c in enumerate(convs):
        conv = keras.layers.Conv2D(lunits[0] // len(convs), (1, c), activation=activation,
                                   use_bias=False, name='spatial_conv_{0}'.format(i),
                                   kernel_regularizer=keras.layers.regularizers.l2(reg),
                                   data_format='channels_last')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.SpatialDropout2D(do/2)(conv)

    for i in range(temp_layers):
        conv = keras.layers.Conv2D(lunits[1], (temporal, 1), activation=activation,
                                   use_bias=False, name='temporal_conv_{0}'.format(i),
                                   kernel_regularizer=keras.layers.regularizers.l2(reg),
                                   data_format='channels_last')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.AveragePooling2D((temp_pool, 1))(conv)
    conv = keras.layers.SpatialDropout2D(do/2)(conv)

    outs = keras.layers.Flatten()(conv)

    for units in lunits[2:]:
        outs = keras.layers.Dense(units, activation=activation,
                                  kernel_regularizer=keras.layers.regularizers.l2(reg))(outs)
        outs = keras.layers.BatchNormalization()(outs)
        outs = keras.layers.Dropout(do)(outs)
    outs = keras.layers.Dense(outputshape, activation='softmax', name='OUT',
                              kernel_regularizer=keras.layers.regularizers.l2(reg))(outs)

    return keras.models.Model(ins, outs)


def RaSCNN(inputshape, outputshape, params=None):

    params = dict() if params is None else params

    ret_seq = bool(params.get('return_sequence', True))
    att_depth = int(params.get('attention_depth', 4))
    attention = int(params.get('attention_units', 76))

    temp_layers = int(params.get('temp_layers', 4))
    steps = int(params.get('steps', 2))
    temporal = int(params.get('temporal', 24))
    temp_pool = int(params.get('temp_pool', 20))
    lunits = [int(x) for x in params.get('lunits', [200, 40])]
    activation = params.get('activation', keras.activations.selu)
    reg = float(params.get('regularization', 0.01))
    do = min(1., max(0., float(params.get('dropout', 0.55))))

    convs = [inputshape[-1]//steps for _ in range(1, steps)]
    convs += [inputshape[-1] - sum(convs) + len(convs)]

    ins = keras.layers.Input(inputshape)

    conv = ExpandLayer()(ins)

    for i, c in enumerate(convs):
        conv = keras.layers.Conv2D(lunits[0]//len(convs), (1, c), activation=activation,
                                   name='spatial_conv_{0}'.format(i),
                                   kernel_regularizer=keras.layers.regularizers.l2(reg))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.SpatialDropout2D(do)(conv)

    for i in range(temp_layers):
        conv = keras.layers.Conv2D(lunits[1], (temporal, 1), activation=activation,
                                   use_bias=False, name='temporal_conv_{0}'.format(i),
                                   kernel_regularizer=keras.layers.regularizers.l2(reg))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.AveragePooling2D((temp_pool, 1,))(conv)
    conv = keras.layers.SpatialDropout2D(do)(conv)
    conv = SqueezeLayer(-2)(conv)

    attn = keras.layers.Bidirectional(AttentionLSTMIn(attention,
                                                      implementation=2,
                                                      # dropout=self.do,
                                                      return_sequences=ret_seq,
                                                      alignment_depth=att_depth,
                                                      style='global',
                                                      # kernel_regularizer=keras.layers.regularizers.l2(self.reg),
                                                      ))(conv)
    conv = keras.layers.BatchNormalization()(attn)

    if ret_seq:
        conv = keras.layers.Flatten()(conv)
    outs = conv
    for units in lunits[2:]:
        outs = keras.layers.Dense(units, activation=activation,
                                  kernel_regularizer=keras.layers.regularizers.l2(reg))(outs)
        outs = keras.layers.BatchNormalization()(outs)
        outs = keras.layers.Dropout(do)(outs)
    outs = keras.layers.Dense(outputshape, activation='softmax')(outs)

    return keras.models.Model(ins, outs)
