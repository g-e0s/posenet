"""Module provides misc utils for model definition and training"""

import tensorflow as tf


class StepDecay:
    """Step decay for learning rate scheduler"""
    def __init__(self, init_lr=0.001, lr_decay_rate=0.1, lr_decay_step=10, min_lr=0.00001):
        self.init_lr = init_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.min_lr = min_lr
        
    def __call__(self, epoch):
        lr = self.init_lr * self.lr_decay_rate ** (epoch // self.lr_decay_step)
        return max(lr, self.min_lr)


class RangeNorm(tf.keras.initializers.Initializer):
    """
    Initializer that generates tensors initialized with columns' norms
    proportional to specified values.
    Arguments
        ranges: np.array; target norms of columns.
    """

    def __init__(self, ranges):
        self.ranges = ranges

    def __call__(self, shape, dtype=None, partition_info=None):
        x = tf.truncated_normal(shape, dtype=tf.float32)
        multiplier = tf.divide(tf.cast(self.ranges, tf.float32), tf.linalg.norm(x, axis=0))
        return tf.transpose(tf.multiply(tf.transpose(x), tf.reshape(multiplier, (-1, 1))))

    def get_config(self):
        return {'ranges': self.ranges}
