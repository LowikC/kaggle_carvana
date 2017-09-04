import os
import cv2
import logging
import argparse
import progressbar
import multiprocessing
import numpy as np
from functools import partial
from utils_file import iter_files_with_ext, get_car_id_rot


def median_worker(car_id, src_dir, dst_dir, dst_size=(240, 160)):
    n_rotations = 16
    blur = 3
    # Load all images, resize and blur
    imgs = np.zeros((n_rotations,) + dst_size[::-1] + (3, ), dtype=np.uint8)
    for rot in range(1, n_rotations + 1):
        basename = car_id + "_{:02d}".format(rot)
        src_filename = os.path.join(src_dir, basename + ".jpg")
        img = cv2.imread(src_filename)
        img = cv2.resize(img, dst_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.medianBlur(img, blur)
        imgs[rot-1, ...] = img
    # Get the median and save it
    median = np.round(np.median(imgs, axis=0)).astype(np.uint8)
    dst_filename = os.path.join(dst_dir, car_id + ".png")
    cv2.imwrite(dst_filename, median)


def median_all(src_dir, src_ext, dst_dir, dst_size):
    os.makedirs(dst_dir, exist_ok=True)
    iter_images = iter_files_with_ext(src_dir, ext=src_ext)
    car_ids = set((get_car_id_rot(fn)[0] for fn in iter_images))

    median_worker_spec = partial(median_worker,
                                 src_dir=src_dir,
                                 dst_dir=dst_dir,
                                 dst_size=dst_size)
    with multiprocessing.Pool() as pool,\
            progressbar.ProgressBar(0, len(car_ids)) as pbar:
        for i, _ in enumerate(
                pool.imap_unordered(median_worker_spec, car_ids)):
            pbar.update(i + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Compute median images')
    parser.add_argument('--src_dir',
                        type=str, default="../data/train",
                        help='Directory containing the images.')
    parser.add_argument('--src_ext',
                        type=str, default=".jpg",
                        help='Extensions of images files in the src dir.')
    parser.add_argument('--dst_dir',
                        type=str, default="../data/train_median",
                        help='Directory to save the median images.')
    parser.add_argument('--target_width',
                        type=int, default=240,
                        help="Target width")
    parser.add_argument('--target_height',
                        type=int, default=160,
                        help="Target height")
    args = parser.parse_args()

    median_all(args.src_dir, args.src_ext, args.dst_dir,
               (args.target_width, args.target_height))
