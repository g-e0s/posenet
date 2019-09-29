
from keras_preprocessing.image import DataFrameIterator
import numpy as np
import skimage


class DFIterator(DataFrameIterator):
    def __init__(self, coordinate_split_index=3, n_auxiliary=0, random_crop_shape=(224, 224), *args, **kwargs):
        self.coordinate_split_index = coordinate_split_index
        self.n_auxiliary = n_auxiliary
        self.random_crop_shape = random_crop_shape
        super().__init__(*args, **kwargs)
        
    def _get_batches_of_transformed_samples(self, index_array):
        x, y = super()._get_batches_of_transformed_samples(index_array)
        x = np.array([self.subtract_mean(self.random_crop(img, self.random_crop_shape)) for img in x])
        y = [y[:, :self.coordinate_split_index], y[:, self.coordinate_split_index:]]
        y = tuple(y*(self.n_auxiliary + 1))
        return x, y
    
    @staticmethod
    def random_crop(img, crop_shape):
        h, w, _ = img.shape
        m, n = crop_shape
        offset_x = np.random.randint(h-m)
        offset_y = np.random.randint(w-n)
        return img[offset_x : offset_x + m, offset_y : offset_y + n]

    @staticmethod
    def subtract_mean(img):
        mean = img.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
        return img - mean

    @staticmethod    
    def proportional_resize(img, size):
        h, w, _ = img.shape
        if h >= w:
            w_new = size
            h_new = int(round(size * h / w))
            resized = skimage.transform.resize(img, (h_new, w_new))
        else:
            h_new = size
            w_new = int(round(size * w / h))
            resized = skimage.transform.resize(img, (h_new, w_new))
        return resized

    @staticmethod
    def center_crop(img):
        h, w, _ = img.shape
        if h >= w:
            offset = (h - w) // 2
            cropped = img[offset : offset+w]
        else:
            offset = (w - h) // 2
            cropped = img[:, offset : offset+h]
        return cropped
