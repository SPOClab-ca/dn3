from layers import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model
from tensorflow.python.keras.constraints import max_norm


def ShallowFBCSP(inputshape, outputshape):
    model = keras.models.Sequential()
    model.add(ExpandLayer(input_shape=inputshape))
    model.add(keras.layers.Conv2D(40, (1, 25), activation='linear', data_format='channels_last'))
    model.add(keras.layers.Conv2D(40, (25, 1), activation='linear', data_format='channels_last'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(tf.square))
    model.add(keras.layers.AveragePooling2D((1, 75), 15))
    model.add(keras.layers.Activation(lambda x: tf.log(tf.maximum(x, tf.constant(1e-6)))))
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


def SCNN(inputshape, outputshape, params=None):

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


def get_dummy_model_tofit(input_size, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=input_size),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def EEGNet(nb_classes, Chans=64, Samples=128,
             dropoutRate=0.5, kernLength=64, F1=8,
             D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    """
    :param nb_classes: int, number of classes to classify
    :param Chans: number of channels in the EEG data
    :param Samples: number of time points in the EEG data
    :param dropoutRate: dropout fraction
    :param kernLength: length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
    :param F1: number of temporal filters (F1)  Default: F1 = 8, F2 = F1 * D.
    :param D:  number of spatial filters to learn within each temporal convolution. Default: D = 2
    :param F2: number of pointwise filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
    :param norm_rate: dropout fraction
    :param dropoutType: Either SpatialDropout2D or Dropout, passed as a string.
    :return: tf.keras.models
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGNet_SSVEP(nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, dropoutType='Dropout'):
    """ SSVEP Variant of EEGNet, as used in [1].
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn.
      D               : number of spatial filters to learn within each temporal
                        convolution.
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.


    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6).
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DenseTCNN(samples_t, channels, targets=4, t_growth=20, do=0.3, subjects=1, temp_layers=(6, 12, 24, 16),
              compress_theta=0.9, num_init_features=40, bn_size=4, data_format='channels_first'):
    if isinstance(temp_layers, int):
       temp_layers = 3 * [temp_layers]

    ins = InputLayer((channels, samples_t))

    # First convolution
    cnn1 = Conv1D(num_init_features, kernel_size=2, strides=1, padding=1, use_bias=False, name="conv0",
                  data_format=data_format)(ins)
    cnn1 = BatchNormalization(axis=1, name='norm0')(cnn1)
    cnn1 = ReLU(name='relu0')(cnn1)
    d_in = MaxPool1D(3, strides=2, data_format=data_format)(cnn1)

    for num_layers in temp_layers[:-1]:
        d_in = dense_block_1d(d_in, num_layers, bn_size, growth_rate=t_growth, drop_rate=do,
                              data_format=data_format)(d_in)
        num_init_features = int(compress_theta * num_init_features)
        d_in = transition(d_in, num_init_features)

    classifier = BatchNormalization(axis=1 if data_format == 'channel_first' else -1, name='features')(d_in)
    classifier = GlobalAveragePooling1D()(classifier)
    classifier = Dense(targets, activation='softmax')(classifier)

    return Model(inputs=ins, outputs=classifier)



