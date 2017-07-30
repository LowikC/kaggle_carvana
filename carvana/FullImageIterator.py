import os
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing.image import Iterator


class FullImageIterator(Iterator):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_ids,
                 batch_size=4,
                 target_shape=(160, 240),
                 crop_shape=None,
                 shuffle=True,
                 seed=42,
                 debug_dir=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.debug_dir = debug_dir
        self.n_indices = len(self.image_ids)
        self.target_shape = target_shape
        self.crop_shape = crop_shape if crop_shape else target_shape
        self.steps_per_epoch = self.n_indices // batch_size
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        super(FullImageIterator, self).__init__(self.n_indices, batch_size,
                                                shuffle, seed)

    def normalize_x(self, x):
        x[..., 0] -= 127
        x[..., 1] -= 127
        x[..., 2] -= 127
        return x

    def distribution(self):
        pos, neg = 0, 0
        for _ in range(self.steps_per_epoch):
            x, y = self.next()
            n_pos = np.count_nonzero(y)
            n_neg = y.ravel().shape[0] - n_pos
            pos += n_pos
            neg += n_neg
        return neg, pos

    def random_transform(self, x, y):
        return x, y

    def resize(self, img, interp):
        if img.shape[:2] != self.target_shape:
            return cv2.resize(img, self.target_shape[:2][::-1],
                              interpolation=interp)
        else:
            return img

    def crop_output(self, mask):
        if self.crop_shape != self.target_shape:
            crop_sx = (self.target_shape[1] - self.crop_shape[1]) // 2
            crop_sy = (self.target_shape[0] - self.crop_shape[0]) // 2
            mask_cropped = mask[crop_sy: crop_sy + self.crop_shape[0],
                                crop_sx: crop_sx + self.crop_shape[1]]
            return mask_cropped
        else:
            return mask

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,
                            self.target_shape[0],
                            self.target_shape[1],
                            3),
                           dtype=np.uint8)
        batch_y = np.zeros((current_batch_size,
                            self.crop_shape[0],
                            self.crop_shape[1],
                            1),
                           dtype=np.uint8)

        # For each index, we load the data and apply needed transformation
        for i, j in enumerate(index_array):
            image_id = self.image_ids[j]
            img_filename = os.path.join(self.image_dir, image_id + ".png")
            mask_filename = os.path.join(self.mask_dir, image_id + "_mask.png")
            img = np.array(Image.open(img_filename))
            mask = np.array(Image.open(mask_filename))

            img = self.resize(img, interp=cv2.INTER_LINEAR)
            mask = self.resize(mask, interp=cv2.INTER_NEAREST)

            img, mask = self.random_transform(img, mask)

            batch_x[i, ...] = img
            batch_y[i, :, :, 0] = mask

        if self.debug_dir:
            for i in range(batch_x.shape[0]):
                img_fn = os.path.join(self.debug_dir, "{:02d}_img.png".format(i))
                mask_fn = os.path.join(self.debug_dir, "{:02d}_mask.png".format(i))
                cv2.imwrite(img_fn, batch_x[i, ...])
                cv2.imwrite(mask_fn, batch_y[i, ...] * 255)

        return self.normalize_x(batch_x), batch_y
