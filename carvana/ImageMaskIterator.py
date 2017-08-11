import os
import numpy as np
import cv2
from PIL import Image
import progressbar
from keras.preprocessing.image import Iterator


class ImageMaskIterator(Iterator):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_ids,
                 batch_size=4,
                 target_shape=(160, 240),
                 crop_shape=None,
                 data_augmentation=None,
                 xpreprocess=None,
                 ypreprocess=None,
                 shuffle=True,
                 seed=42,
                 debug_dir=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.debug_dir = debug_dir
        if data_augmentation is None:
            self.data_augmentation = lambda x, y: (x, y)
        else:
            self.data_augmentation = data_augmentation
        if xpreprocess is None:
            self.xpreprocess = lambda x: x
        else:
            self.xpreprocess = xpreprocess

        if ypreprocess is None:
            self.ypreprocess = lambda x: x
        else:
            self.ypreprocess = ypreprocess
        self.n_indices = len(self.image_ids)
        self.target_shape = target_shape
        self.crop_shape = crop_shape if crop_shape else target_shape
        self.steps_per_epoch = int(np.ceil(self.n_indices / batch_size))
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        super(ImageMaskIterator, self).__init__(self.n_indices,
                                                            batch_size,
                                                            shuffle, seed)

    def distribution(self):
        class_counts = np.zeros((4,), dtype=np.int64)
        with progressbar.ProgressBar(0, self.steps_per_epoch) as pbar:
            for i in range(self.steps_per_epoch):
                x, y = self.next()
                class_ids, counts = np.unique(y, return_counts=True)
                for cls, cnt in zip(class_ids, counts):
                    class_counts[cls] += cnt
                pbar.update(i + 1)
        return class_counts

    def random_transform(self, x, y):
        if not self.data_augmentation:
            return x, y
        return x, y

    def check_size(self, img):
        if img.shape[:2] != self.target_shape:
            raise Exception("Wrong size")

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
            img = np.array(Image.open(img_filename))
            self.check_size(img)
            if self.mask_dir:
                mask_filename = os.path.join(self.mask_dir, image_id + "_mask.png")
                mask = np.array(Image.open(mask_filename))
                self.check_size(mask)
            else:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

            mask = mask.reshape(mask.shape + (1, ))
            img, mask = self.data_augmentation(img, mask)

            batch_x[i, ...] = img
            batch_y[i, ...] = mask

        if self.debug_dir:
            for i in range(batch_x.shape[0]):
                img_fn = os.path.join(self.debug_dir, "{:02d}_img.png".format(i))
                mask_fn = os.path.join(self.debug_dir, "{:02d}_mask.png".format(i))
                cv2.imwrite(img_fn, batch_x[i, ...])
                cv2.imwrite(mask_fn, batch_y[i, ...])
        return self.xpreprocess(batch_x), self.ypreprocess(batch_y)