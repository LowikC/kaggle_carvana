import os
import logging
import argparse
import rle
import cv2
from utils_file import num_lines
import numpy as np
import shutil
import progressbar


def check_submission(args):
    os.makedirs(args.dst_dir, exist_ok=True)

    n_samples = num_lines(args.submission) - 1
    if args.num < 0:
        selected_samples_idx = set(range(n_samples))
    else:
        selected_samples_idx = set(np.random.permutation(n_samples)[:args.num])

    logging.info("Checking the submission ({} samples)".format(n_samples))
    with open(args.submission, "r") as csv_file,\
            progressbar.ProgressBar(0, n_samples) as pbar:
        header = next(csv_file)
        if header.strip() != "img,rle_mask":
            logging.error("Wrong header: {}".format(header))
            return
        for idx, line in enumerate(csv_file):
            if idx in selected_samples_idx:
                filename, rle_code = line.split(",")
                mask = rle.loads(rle_code)
                uid, _ = os.path.splitext(filename)
                cv2.imwrite(os.path.join(args.dst_dir, uid + ".png"), mask * 255)
                shutil.copy(os.path.join(args.images_dir, filename),
                            os.path.join(args.dst_dir, filename))
            pbar.update(idx + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Read a submission file and '
                    'create a directory with some images and their mask.')
    parser.add_argument('--dst_dir',
                        type=str, default="../data/check",
                        help='Directory to save the predicted masks and images.')
    parser.add_argument('--submission',
                        type=str,
                        help="Path to the submission file.")
    parser.add_argument('--images_dir',
                        type=str,
                        help="Path to the images directory.")
    parser.add_argument('--num',
                        type=int, default=10,
                        help="Number of samples to stored."
                             "If -1, store all samples in the submission.")
    args = parser.parse_args()

    check_submission(args)
