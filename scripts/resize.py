from PIL import Image
import os
import multiprocessing
from functools import partial
import logging
import progressbar
from utils_file import iter_files_with_ext


def resize_worker(src_filename, dst_dir, target_size, is_boolean=False):
    try:
        img = Image.open(src_filename)
        resample_method = Image.NEAREST if is_boolean else Image.BICUBIC
        img_resized = img.resize(target_size, resample=resample_method)

        src_basename = os.path.basename(src_filename)
        dst_basename, _ = os.path.splitext(src_basename)
        dst_filename = os.path.join(dst_dir, dst_basename + ".png")
        img_resized.save(dst_filename)
    except Exception as e:
        logging.error("Can't process image {}: {}".format(src_filename, e))


def resize_all(src_dir, src_ext, dst_dir, target_size, is_boolean=False):
    os.makedirs(dst_dir, exist_ok=True)
    iter_images = iter_files_with_ext(src_dir, ext=src_ext)

    resize_worker_spec = partial(resize_worker, dst_dir=dst_dir,
                                 target_size=target_size, is_boolean=is_boolean)
    with multiprocessing.Pool() as pool,\
            progressbar.ProgressBar(max_value=progressbar.UnknownLength) as pbar:
        for i, _ in enumerate(
                pool.imap_unordered(resize_worker_spec, iter_images)):
            pbar.update(i + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resize_all("../data/train/", ".jpg", "../data/train_240x160/", (240, 160))