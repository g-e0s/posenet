"""A module provides image data generator"""

from keras_preprocessing.image import DataFrameIterator
import numpy as np
from tensorflow.keras.utils import Sequence


class RandomCropGenerator(DataFrameIterator, Sequence):
    """
    A DataFrameIterator class extension to generate randomly cropped image samples
    with corresponding location coordinates
    Arguments:
        coordinate_split_index: an index to split location and orientation coordinates
        n_auxiliary: number of auxiliary targets to generate
                     (useful for models using auxiliary losses)
        random_crop_shape: a tuple containing height and width
                           of a random crop sampled from an image
    """
    def __init__(self, coordinate_split_index=3, n_auxiliary=0, random_crop_shape=(224, 224),
                 *args, **kwargs):
        self.coordinate_split_index = coordinate_split_index
        self.n_auxiliary = n_auxiliary
        self.random_crop_shape = random_crop_shape
        super().__init__(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
        batch_x = np.array([self.subtract_mean(self.random_crop(img, self.random_crop_shape))
                            for img in batch_x])
        batch_y = [batch_y[:, :self.coordinate_split_index],
                   batch_y[:, self.coordinate_split_index:]]
        batch_y = tuple(batch_y*(self.n_auxiliary + 1))
        return batch_x, batch_y

    @staticmethod
    def random_crop(img, crop_shape):
        """Produces random image crop of given shape"""
        img_h, img_w, _ = img.shape
        crop_h, crop_w = crop_shape
        offset_x = np.random.randint(img_h - crop_h)
        offset_y = np.random.randint(img_w - crop_w)
        return img[offset_x : offset_x + crop_h, offset_y : offset_y + crop_w]

    @staticmethod
    def subtract_mean(img):
        """Subtracts channel means from an image"""
        mean = img.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
        return img - mean
