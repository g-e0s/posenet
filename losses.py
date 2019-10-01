"""Module provides losses and metrics for posenet"""

import math
import tensorflow as tf


def euclidean_normalized(target, output):
    """Euclidean distance between normalized vectors"""
    true_orientation = tf.div_no_nan(target, tf.linalg.norm(target, axis=-1, keepdims=True))
    predicted_orientation = tf.div_no_nan(output, tf.linalg.norm(output, axis=-1, keepdims=True))
    return tf.linalg.norm(true_orientation - predicted_orientation, axis=-1)

def euclidean(target, output):
    """Euclidean distance"""
    return tf.linalg.norm(target - output, axis=-1)

def orientation_angle(target, output):
    """Angle between orientations given by quaternions"""
    true_orientation = tf.div_no_nan(target, tf.linalg.norm(target, axis=-1, keepdims=True))
    predicted_orientation = tf.div_no_nan(output, tf.linalg.norm(output, axis=-1, keepdims=True))
    dot = tf.reduce_sum(tf.multiply(true_orientation, predicted_orientation), axis=-1)
    dot = tf.math.minimum(tf.math.maximum(dot, -1.0), 1.0)
    angle = tf.math.acos(2 * tf.square(dot) - 1) * 180 / math.pi
    return angle
