import os
import logging
import argparse
import json
import cv2
import progressbar
import numpy as np
from keras.models import load_model
from skimage.transform import resize
import rle
from utils_file import iter_files_with_ext
from FullImageIterator import FullImageIterator
import constants

def get_model_uid(model_filename):
    model_uid = os.path.basename(model_filename)
    model_uid, _ = os.path.splitext(model_uid)
    return model_uid


def get_tmp_raw(args):
    model_uid = get_model_uid(args.model)
    tmp_dir = args.tmp_dir
    return os.path.join(tmp_dir, model_uid, "raw")


def get_tmp_scaled(args):
    model_uid = get_model_uid(args.model)
    tmp_dir = args.tmp_dir
    return os.path.join(tmp_dir, model_uid, "scaled")


def get_submission_filename(args):
    model_uid = get_model_uid(args.model)
    tmp_dir = args.tmp_dir
    return os.path.join(tmp_dir, model_uid, "submission.csv")


def predict_test(args):
    """
    Compute the prediction mask for all test images.
    :param args: 
    """
    model = load_model(args.model, compile=False)
    raw_dir = get_tmp_raw(args)
    os.makedirs(raw_dir, exist_ok=True)
    input_shape = model.input_shape[1:3]
    with open(args.test, "r") as jfile:
        test_ids = json.load(jfile)
    logging.info("Apply prediction model on {} images".format(len(test_ids)))
    test_iterator = FullImageIterator(args.images_dir, None, test_ids,
                                      batch_size=args.batch_size,
                                      target_shape=input_shape,
                                      shuffle=False)

    current_sample = 0
    with progressbar.ProgressBar(0, len(test_ids)) as pbar:
        for batch_idx in range(test_iterator.steps_per_epoch):
            bx, _ = next(test_iterator)
            bmasks = model.predict_on_batch(bx)
            for i in range(bmasks.shape[0]):
                if current_sample < len(test_ids):
                    out_basename = "{}.npz".format(test_ids[current_sample])
                    out_filename = os.path.join(raw_dir, out_basename)
                    np.savez_compressed(out_filename, mask=bmasks[i, ...])
                    current_sample += 1
                    pbar.update(current_sample)


def scale_test(args):
    raw_dir = get_tmp_raw(args)
    npz_files = list(iter_files_with_ext(raw_dir, ".npz"))
    logging.info("Scale {} predictions...".format(len(npz_files)))

    scaled_dir = get_tmp_scaled(args)
    os.makedirs(scaled_dir, exist_ok=True)
    with progressbar.ProgressBar(0, len(npz_files)) as pbar:
        for i, fn in enumerate(npz_files):
            mask = np.load(fn)["mask"]
            mask_full_resolution = resize(mask, constants.default_shape,
                                          mode='constant', order=5)
            mask_full_resolution = (mask_full_resolution > 0.5).astype(np.uint8)
            basename = os.path.basename(fn)
            basename, _ = os.path.splitext(basename)
            basename += ".png"
            out_filename = os.path.join(scaled_dir, basename)
            cv2.imwrite(out_filename, mask_full_resolution)
            pbar.update(i + 1)


def write_submission(args):
    """
    Read predicted mask and write in submission format.
    :param args: 
    """
    scaled_dir = get_tmp_scaled(args)
    os.makedirs(scaled_dir, exist_ok=True)

    scaled_files = list(iter_files_with_ext(scaled_dir, ".png"))
    logging.info("Write submission for {} masks...".format(len(scaled_files)))

    csv_filename = get_submission_filename(args)
    with open(csv_filename, "w") as csv_file:
        csv_file.write("img,rle_mask\n")
        with progressbar.ProgressBar(0, len(scaled_files)) as pbar:
            for i, fn in enumerate(scaled_files):
                mask = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                basename = os.path.basename(fn)
                basename, _ = os.path.splitext(basename)
                csv_file.write(basename + ".jpg" + ",")
                csv_file.write(rle.dumps(mask))
                csv_file.write("\n")
                pbar.update(i + 1)


def make_submission(args):
    model_uid = get_model_uid(args.model)
    logging.info("Model uid: {}".format(model_uid))

    if not args.use_tmp:
        logging.info("Don't use tmp directory, predict masks from model.")
        predict_test(args)
        scale_test(args)
    else:
        logging.info("Use tmp directory: {}".format(get_tmp_scaled(args)))
    write_submission(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Create a submission given a trained model.')
    parser.add_argument('--tmp_dir',
                        type=str, default="../data/tmp",
                        help='Temp directory to save the predicted masks.')
    parser.add_argument('--use_tmp',
                        action='store_true',
                        help="Don't predict, use directly the predictions in the tmp directory.")
    parser.add_argument('--model',
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument('--images_dir',
                        type=str,
                        help="Path to the images directory.")
    parser.add_argument('--test',
                        type=str,
                        help="Path to the json file with the test ids.")
    parser.add_argument('--batch_size',
                        type=int, default=4,
                        help="Path to the json file with the test ids.")
    args = parser.parse_args()

    make_submission(args)
