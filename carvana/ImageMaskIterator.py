import os
import numpy as np
import cv2
from PIL import Image
import progressbar
from collections import defaultdict
from keras.preprocessing.image import Iterator


class ImageMaskIterator(Iterator):
    def __init__(self,
                 image_dir,
                 image_ids,
                 mask_dir="",
                 image_ext=".png",
                 mask_ext=".png",
                 batch_size=4,
                 target_shape=(160, 240),
                 crop_shape=None,
                 n_patches_per_image=1,
                 data_augmentation=None,
                 xpreprocess=None,
                 ypreprocess=None,
                 shuffle=True,
                 seed=42,
                 debug_dir=None):
        """
        Iterator on images and masks samples in a given directory.
        :param image_dir: Directory containing the images.
        :param image_ids: Ids of the images to use.
        :param mask_dir: Directory containing the masks. Can be empty.
        :param image_ext: Extension of the image files (default: .png)
        :param mask_ext: Extension of the mask files (default: .png)
        :param batch_size: Number of samples per batch.
        :param target_shape: Size of the X samples.
        :param crop_shape: Size of the Y samples.
        :param n_patches_per_images: Number of patches sampled on one image.
            batch_size must be divisible by n_patches_per_images.
        :param data_augmentation: Function for data augmentation
            (img, mask) -> (augmented img, augmented mask).
        :param xpreprocess: Function applied to preprocess the X samples
            (batch img) -> (preprocessed batch img)
        :param ypreprocess: Function applied to preprocess the Y samples
            (batch mask) -> (preprocessed batch mask)
        :param shuffle: Shuffle the images.
            Otherwise, samples are returned in the same order than images_ids
        :param seed: Seed for shuffling.
        :param debug_dir: If the directory exist,
            save X and Y samples before preprocessing.
        """
        self.image_ids = image_ids
        self.n_patches_per_image = n_patches_per_image
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
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
        self.n_indices = len(self.image_ids) * self.n_patches_per_image
        self.target_shape = target_shape
        self.crop_shape = crop_shape if crop_shape else target_shape
        self.steps_per_epoch = int(np.ceil(self.n_indices / batch_size))
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        if seed is not None:
            np.random.seed(seed)
        super(ImageMaskIterator, self).__init__(self.n_indices // n_patches_per_image,
                                                batch_size // n_patches_per_image,
                                                shuffle, seed)

    def distribution(self, n_batches=None):
        """
        Compute the class distribution of the output.
        :return: A dict class_id -> class_cnt
        """
        if n_batches is None:
            n_batches = self.steps_per_epoch // self.n_patches_per_image
        class_counts = defaultdict(int)
        with progressbar.ProgressBar(0, n_batches) as pbar:
            for i in range(n_batches):
                x, y = self.next()
                class_ids, counts = np.unique(y, return_counts=True)
                for cls, cnt in zip(class_ids, counts):
                    class_counts[cls] += cnt
                pbar.update(i + 1)
        return class_counts

    def crop_output(self, mask):
        if self.crop_shape != self.target_shape:
            crop_sx = (self.target_shape[1] - self.crop_shape[1]) // 2
            crop_sy = (self.target_shape[0] - self.crop_shape[0]) // 2
            mask_cropped = mask[crop_sy: crop_sy + self.crop_shape[0],
                                crop_sx: crop_sx + self.crop_shape[1]]
            return mask_cropped
        else:
            return mask

    def valid_patch(self, mask_patch):
        n_foreground = np.count_nonzero(mask_patch)
        n_total = self.target_shape[0] * self.target_shape[1]
        return 0.1 < n_foreground / n_total < 0.9

    def sample(self, img, mask, n_patches):
        """
        Sample randomly n_patches from img and mask, with data augmentation.
        """
        batch_x = np.zeros((n_patches,
                            self.target_shape[0],
                            self.target_shape[1],
                            3),
                           dtype=np.uint8)
        batch_y = np.zeros((n_patches,
                            self.crop_shape[0],
                            self.crop_shape[1],
                            1),
                           dtype=np.uint8)

        img, mask = self.data_augmentation(img, mask)

        if img.shape[:2] == self.target_shape:
            for i in range(n_patches):
                img_aug, mask_aug = self.data_augmentation(img, mask)
                batch_x[i, ...] = img_aug
                batch_y[i, :, :, 0] = self.crop_output(mask_aug)
        else:
            n_patches_selected = 0
            while n_patches_selected < n_patches:
                sy_max = img.shape[0] - self.target_shape[0]
                sx_max = img.shape[1] - self.target_shape[1]
                sy = np.random.randint(0, sy_max + 1)
                sx = np.random.randint(0, sx_max + 1)
                mask_patch = mask[sy: sy + self.target_shape[0],
                                  sx: sx + self.target_shape[1]]
                if self.valid_patch(mask_patch):
                    img_patch = img[sy: sy + self.target_shape[0],
                                    sx: sx + self.target_shape[1], :]
                    img_patch_aug, mask_patch_aug = self.data_augmentation(img_patch, mask_patch)
                    batch_x[n_patches_selected, ...] = img_patch_aug
                    batch_y[n_patches_selected, :, :, 0] = mask_patch_aug
                    n_patches_selected += 1
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size * self.n_patches_per_image,
                            self.target_shape[0],
                            self.target_shape[1],
                            3),
                           dtype=np.uint8)
        batch_y = np.zeros((current_batch_size * self.n_patches_per_image,
                            self.crop_shape[0],
                            self.crop_shape[1],
                            1),
                           dtype=np.uint8)

        # For each index, we load the data and apply needed transformation
        for i, j in enumerate(index_array):
            image_id = self.image_ids[j]
            img_filename = os.path.join(self.image_dir, image_id + self.image_ext)
            img = np.array(Image.open(img_filename))
            if self.mask_dir:
                mask_filename = os.path.join(self.mask_dir, image_id + "_mask" + self.mask_ext)
                mask = np.array(Image.open(mask_filename))
            else:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

            mbatch_x, mbatch_y = self.sample(img, mask, self.n_patches_per_image)
            sidx = i * self.n_patches_per_image
            eidx = (i + 1) * self.n_patches_per_image
            batch_x[sidx: eidx, ...] = mbatch_x
            batch_y[sidx: eidx, ...] = mbatch_y

        if self.debug_dir:
            for i in range(batch_x.shape[0]):
                img_fn = os.path.join(self.debug_dir, "{:02d}_img.png".format(i))
                mask_fn = os.path.join(self.debug_dir, "{:02d}_mask.png".format(i))
                cv2.imwrite(img_fn, batch_x[i, ...])
                cv2.imwrite(mask_fn, batch_y[i, :, :, 0] * 60)
        return self.xpreprocess(batch_x), self.ypreprocess(batch_y)
