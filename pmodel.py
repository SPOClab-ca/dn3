import tensorflow as tf


class Pmodel(tf.keras.models.Model):

    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
        pass

    def compute_output_signature(self, input_signature):
        pass


def get_dummy_model_tofit(input_size, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=input_size),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model
