import argparse
from PIL import Image
import os
import multiprocessing
from functools import partial
import logging
import progressbar
import numpy as np
import cv2
from utils_file import iter_files_with_ext
import constants


def resize_worker(src_filename, dst_dir, target_size, is_boolean=False):
    try:
        img = Image.open(src_filename)
        if is_boolean:
            img_1920x1280 = np.zeros((1280, 1920), dtype=np.uint8)
            img_1920x1280[:, 1:1919] = np.array(img) * 255
            img_1920x1280 = Image.fromarray(img_1920x1280)
            img_resized = img_1920x1280.resize(target_size, resample=Image.BILINEAR)
            img_resized = Image.fromarray((np.array(img_resized) > 127).astype(np.uint8))
        else:
            img_1920x1280 = np.zeros((1280, 1920, 3), dtype=np.uint8)
            img_1920x1280[:, 1:1919, :] = np.array(img)
            img_resized = cv2.resize(img_1920x1280, target_size, interpolation=cv2.INTER_AREA)
            img_resized = Image.fromarray(img_resized)

        src_basename = os.path.basename(src_filename)
        dst_basename, _ = os.path.splitext(src_basename)
        dst_filename = os.path.join(dst_dir, dst_basename + ".png")
        img_resized.save(dst_filename)
    except Exception as e:
        logging.error("Can't process image {}: {}".format(src_filename, e))


def resize_all(src_dir, src_ext, dst_dir, target_size, is_boolean=False):
    os.makedirs(dst_dir, exist_ok=True)
    list_images = list(iter_files_with_ext(src_dir, ext=src_ext))

    resize_worker_spec = partial(resize_worker, dst_dir=dst_dir,
                                 target_size=target_size, is_boolean=is_boolean)
    with multiprocessing.Pool() as pool,\
            progressbar.ProgressBar(0, len(list_images)) as pbar:
        for i, _ in enumerate(
                pool.imap_unordered(resize_worker_spec, list_images)):
            pbar.update(i + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Create a submission given a trained model.')
    parser.add_argument('--src_dir',
                        type=str, default="../data/train",
                        help='Directory containing the images.')
    parser.add_argument('--src_ext',
                        type=str, default=".jpg",
                        help='Extensions of images files in the src dir.')
    parser.add_argument('--dst_dir',
                        type=str, default="../data/train_240x160",
                        help='Directory to save the resized images.')
    parser.add_argument('--target_width',
                        type=int, default=240,
                        help="Target width")
    parser.add_argument('--target_height',
                        type=int, default=160,
                        help="Target height")
    parser.add_argument('--is_boolean',
                        action="store_true",
                        help="Boolean image in input")
    args = parser.parse_args()

    resize_all(args.src_dir, args.src_ext, args.dst_dir,
               (args.target_width, args.target_height), args.is_boolean)
