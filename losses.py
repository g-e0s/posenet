import math
import tensorflow as tf


def euclidean_normalized(target, output):
    true_orientation = tf.div_no_nan(target, tf.linalg.norm(target, axis=-1, keepdims=True))
    predicted_orientation = tf.div_no_nan(output, tf.linalg.norm(output, axis=-1, keepdims=True))
    return tf.linalg.norm(true_orientation - predicted_orientation, axis=-1)

def euclidean(target, output):
    return tf.linalg.norm(target - output, axis=-1)

def orientation_angle(target, output):
    true_orientation = tf.div_no_nan(target, tf.linalg.norm(target, axis=-1, keepdims=True))
    predicted_orientation = tf.div_no_nan(output, tf.linalg.norm(output, axis=-1, keepdims=True))
    d = tf.abs(tf.reduce_sum(tf.multiply(true_orientation, predicted_orientation), axis=-1, keepdims=True))
    theta = 2 * tf.math.acos(d) * 180 / math.pi
    return theta