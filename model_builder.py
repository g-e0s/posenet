"""A module provides model building and training functionality"""

import tensorflow as tf
from losses import euclidean, euclidean_normalized, orientation_angle
from utils import RangeNorm, StepDecay


class ModelBuilder:
    """A class for model definition and training"""
    def build_model(self, image_size=(224, 224), pose_regressor_size=1024,
                    loc_size=3, orient_size=4,
                    position_ranges=None, freeze_layers=False):
        """
        Builds posenet model from Xception trained on Imagenet
        Arguments:
            image_size: a size of input images
            pose_regressor_size: a number of units in pose regressor layer
            loc_size: number of location outputs
            orient_size: number of orientation outputs
            position_ranges: a numpy array with spatial ranges
                             of location and orientation dimensions
            freeze_layers: indicates whether all the layers except pose regressor
                           should be nontrainable
        Returns:
            an instance of tensorflow.keras.Model
        """

        model = tf.keras.applications.xception.Xception(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=image_size + (3,))
        # freeze layers
        if freeze_layers:
            for layer in model.layers:
                layer.trainable = False

        # regressor initializers
        if position_ranges is not None:
            init_xyz = RangeNorm(position_ranges[:loc_size])
            init_wpqr = RangeNorm(position_ranges[loc_size:])
        else:
            init_xyz = tf.keras.initializers.TruncatedNormal(stddev=0.5)
            init_wpqr = tf.keras.initializers.TruncatedNormal(stddev=0.05)

        # add pose regressor
        avgpool = tf.keras.layers.GlobalAvgPool2D(name='pose_pool')(model.output)
        pose_dense = tf.keras.layers.Dense(pose_regressor_size, activation="relu", name='pose_dense')(avgpool)
        pose_dense = tf.keras.layers.Dropout(rate=0.5, name='pose_dropout')(pose_dense)
        pose_xyz = tf.keras.layers.Dense(loc_size, activation="linear", name='xyz',
                                         kernel_initializer=init_xyz,
                                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(pose_dense)
        pose_wpqr = tf.keras.layers.Dense(orient_size, activation="tanh", name='wpqr',
                                          kernel_initializer=init_wpqr,
                                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(pose_dense)

        # create final model
        model = tf.keras.Model(name='posenet', inputs=model.input,
                               outputs=[pose_xyz, pose_wpqr])
        return model

    def fit_model(self, model, train_generator, validation_generator=None,
                  epochs=30, lr=0.001, lr_decay_rate=0.1, lr_decay_step=10, min_lr=0.00001,
                  beta=20):
        """
        Compiles and fits posenet model
        Arguments:
            model: tf.keras.Model instance
            train_generator: an instance of data_generator.RandomCropGenerator
                             for training examples
            validation_generator: an instance of data_generator.RandomCropGenerator
                                  for validation examples
            epochs: number of training epochs
            lr: initial learning rate for an optimizer
            lr_decay_rate: learning rate decay
            lr_decay_step: number of epochs between learning rate step decays
            min_lr: minimal learning rate for an optimizer
            beta: a weight applied for orientation loss
        """

        # compile model
        model.compile(
            loss={'xyz': euclidean, 'wpqr': euclidean_normalized},
            loss_weights={'xyz': 1, 'wpqr': beta},
            metrics={'wpqr': orientation_angle},
            optimizer=tf.keras.optimizers.Adam(lr=lr))

        # learning rate scheduler
        lr_schedule = StepDecay(lr, lr_decay_rate, lr_decay_step, min_lr)
        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)]

        # fit model
        model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator,
                            callbacks=callbacks)
