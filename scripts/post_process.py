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


def post_process_worker(src_filename, dst_dir):
    try:
        mask = np.array(Image.open(src_filename))
        r, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        biggest_label = np.argsort(stats[1:, -1], axis=-1)[::-1][0] + 1
        mask_cleaned = (labels == biggest_label).astype(np.uint8)
        mask_cleaned = Image.fromarray(mask_cleaned)

        src_basename = os.path.basename(src_filename)
        dst_basename, _ = os.path.splitext(src_basename)
        dst_filename = os.path.join(dst_dir, dst_basename + ".png")
        mask_cleaned.save(dst_filename)
    except Exception as e:
        logging.error("Can't process image {}: {}".format(src_filename, e))


def post_process_all(src_dir, src_ext, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    list_images = list(iter_files_with_ext(src_dir, ext=src_ext))

    post_worker_spec = partial(post_process_worker, dst_dir=dst_dir)
    with multiprocessing.Pool() as pool,\
            progressbar.ProgressBar(0, len(list_images)) as pbar:
        for i, _ in enumerate(
                pool.imap_unordered(post_worker_spec, list_images)):
            pbar.update(i + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Post process the predicted masks.')
    parser.add_argument('--src_dir',
                        type=str, default="../data/tmp/",
                        help='Directory containing the original predicted masks.')
    parser.add_argument('--src_ext',
                        type=str, default=".png",
                        help='Extensions of images files in the src dir.')
    parser.add_argument('--dst_dir',
                        type=str, default="../data/tmp/postprocess",
                        help='Directory to save the postprocessed masks.')
    args = parser.parse_args()

    post_process_all(args.src_dir, args.src_ext, args.dst_dir)
