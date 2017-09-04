import os
import cv2
import json
import logging
import argparse
import numpy as np
from utils_file import get_car_id_rot, iter_files_with_ext
import multiprocessing
import progressbar
from functools import partial


def get_foreground_bbox(img, median, threshold=30):
    hr, wr, _ = median.shape
    h, w, _ = img.shape

    img_resized = cv2.resize(img, (wr, hr), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.medianBlur(img_resized, 3)

    diff = np.abs(img_resized.astype(np.int32) - median.astype(np.int32))
    diff_rgb = np.sum(diff, axis=-1)
    mask = diff_rgb > threshold
    y, x = np.nonzero(mask)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    s = w / wr
    xmin = np.round(xmin * s)
    xmax = np.round(xmax * s)
    ymin = np.round(ymin * s)
    ymax = np.round(ymax * s)

    return (xmin, xmax), (ymin, ymax)


def foreground_worker(img_filename, median_dir, threshold=30):
    img = cv2.imread(img_filename)
    car_id, _ = get_car_id_rot(img_filename)

    median_filename = os.path.join(median_dir, car_id + ".png")
    median = cv2.imread(median_filename)

    bbox = get_foreground_bbox(img, median, threshold)
    return img_filename, bbox


def foreground_all(src_dir, src_ext, median_dir, dst_filename, threshold):
    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    list_images = list(iter_files_with_ext(src_dir, ext=src_ext))
    fg_worker_spec = partial(foreground_worker,
                             median_dir=median_dir,
                             threshold=threshold)
    results = []
    with multiprocessing.Pool() as pool,\
            progressbar.ProgressBar(0, len(list_images)) as pbar:
        for i, r in enumerate(
                pool.imap_unordered(fg_worker_spec, list_images)):
            results.append(r)
            pbar.update(i + 1)

    bbox_by_fn = dict()
    for fn, bbox in results:
        bbox_by_fn[fn] = list(bbox)

    with open(dst_filename, "w") as jfile:
        json.dump(bbox_by_fn, jfile)


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
    parser.add_argument('--dst',
                        type=str, default="../data/train_foreground.json",
                        help='Path to destination file.')
    parser.add_argument('--median_dir',
                        type=str, default="../data/train_median",
                        help='Directory containing the median images.')
    parser.add_argument('--threshold',
                        type=int, default=30,
                        help="Threshold to detect the foreground.")
    parser.add_argument('--margin',
                        type=float, default=0.1,
                        help="Margin applied on the final bbox")
    args = parser.parse_args()

    foreground_all(args.src_dir, args.src_ext, args.median_dir,
                   args.dst, args.threshold)
