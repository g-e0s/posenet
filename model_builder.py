import tensorflow as tf
from data_generator import DFIterator
from losses import euclidean, euclidean_normalized, orientation_angle


class RangeNorm(tf.keras.initializers.Initializer):
    """Initializer that generates tensors initialized to a constant value.
    # Arguments
        value: float; the value of the generator tensors.
    """

    def __init__(self, ranges):
        self.ranges = ranges

    def __call__(self, shape, dtype=None, partition_info=None):
        x = tf.truncated_normal(shape, dtype=tf.float32)
        multiplier = tf.divide(tf.cast(self.ranges, tf.float32), tf.linalg.norm(x, axis=0))
        return tf.transpose(tf.multiply(tf.transpose(x), tf.reshape(multiplier, (-1, 1))))

    def get_config(self):
        return {'ranges': self.ranges}


class ModelBuilder:
    def build_model(self, image_size=(224, 224), pose_regressor_size=1024, loc_size=3, orient_size=4, position_ranges=None):
        model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=image_size + (3,))
        # freeze layers
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
        x = model.output
        x = tf.keras.layers.GlobalAvgPool2D(name='pose_pool')(x)
        x = tf.keras.layers.Dense(pose_regressor_size, activation="relu", name='pose_dense')(x)
        x = tf.keras.layers.Dropout(rate=0.5, name='pose_dropout')(x)
        position_xyz = tf.keras.layers.Dense(loc_size, activation="linear", name='xyz',
            kernel_initializer=init_xyz,
            kernel_regularizer=tf.keras.regularizers.l2(1.0))(x)
        position_wpqr = tf.keras.layers.Dense(orient_size, activation="tanh", name='wpqr',
            kernel_initializer=init_wpqr,
            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        # create final model 
        model = tf.keras.Model(name='posenet', inputs=model.input, outputs=[position_xyz, position_wpqr])
        return model

    def compile_model(self, model: tf.keras.Model, lr: float, momentum: float, beta: float):
        model.compile(
            #loss={'xyz': euclidean, 'wpqr': euclidean},
            loss={'xyz': euclidean, 'wpqr': euclidean_normalized},
            loss_weights={'xyz': 1, 'wpqr': beta},
            metrics={'wpqr': orientation_angle},
            optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum))

    def pretrain_pose_regressor_layers(self, model: tf.keras.Model, train_generator: DFIterator, lr: float, epochs: int):
        for layer in model.layers:
            layer.trainable = False
        
        
        
