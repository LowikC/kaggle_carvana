import os
import logging
import argparse
import json
import cv2
import progressbar
import numpy as np
from keras.models import load_model
import rle
from unet import preprocess
from ImageMaskIterator import FullImageWithContoursIterator
from PIL import Image


def get_model_uid(model_filename):
    model_uid = os.path.basename(model_filename)
    model_uid, _ = os.path.splitext(model_uid)
    return model_uid


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
    scaled_dir = get_tmp_scaled(args)
    os.makedirs(scaled_dir, exist_ok=True)
    csv_filename = get_submission_filename(args)

    model = load_model(args.model, compile=False)
    input_shape = model.input_shape[1:3]
    with open(args.test, "r") as jfile:
        test_ids = json.load(jfile)
    logging.info("Apply prediction model on {} images".format(len(test_ids)))
    test_iterator = FullImageWithContoursIterator(args.images_dir, None,
                                                  test_ids,
                                                  batch_size=args.batch_size,
                                                  target_shape=input_shape,
                                                  shuffle=False,
                                                  xpreprocess=preprocess)

    current_sample = 0
    with progressbar.ProgressBar(0, len(test_ids)) as pbar, \
            open(csv_filename, "w") as csv_file:
        csv_file.write("img,rle_mask\n")

        for batch_idx in range(test_iterator.steps_per_epoch):
            bx, _ = next(test_iterator)
            bmasks = model.predict_on_batch(bx)

            for i in range(bmasks.shape[0]):
                if current_sample < len(test_ids):
                    mask_full_res = scale(bmasks[i, ...])
                    basename = test_ids[current_sample]
                    # Save predicted mask
                    output_filename = os.path.join(scaled_dir, basename + ".png")
                    cv2.imwrite(output_filename, mask_full_res)
                    # Write in csv file
                    csv_file.write(basename + ".jpg" + ",")
                    csv_file.write(rle.dumps(mask_full_res))
                    csv_file.write("\n")
                    current_sample += 1
                    pbar.update(current_sample)


def load_test(args):
    src_dir = args.src_dir
    csv_filename = get_submission_filename(args)

    with open(args.test, "r") as jfile:
        test_ids = json.load(jfile)

    with progressbar.ProgressBar(0, len(test_ids)) as pbar, \
            open(csv_filename, "w") as csv_file:
        csv_file.write("img,rle_mask\n")

        for i, test_id in enumerate(test_ids):
            mask_filename = os.path.join(src_dir, test_id + ".png")
            mask = np.array(Image.open(mask_filename))
            # Write in csv file
            csv_file.write(test_id + ".jpg" + ",")
            csv_file.write(rle.dumps(mask))
            csv_file.write("\n")
            pbar.update(i + 1)


def scale(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8_full = cv2.resize(mask_u8, (1920, 1280),
                              interpolation=cv2.INTER_CUBIC)
    return (mask_u8_full[:, 1:1919] > 127).astype(np.uint8)


def make_submission(args):
    if os.path.isdir(args.src_dir):
        logging.info("Use existing predicted masks.")
        load_test(args)
    else:
        model_uid = get_model_uid(args.model)
        logging.info("Model uid: {}".format(model_uid))
        predict_test(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Create a submission given a trained model.')
    parser.add_argument('--tmp_dir',
                        type=str, default="../data/tmp",
                        help='Temp directory to save the predicted masks.')
    parser.add_argument('--src_dir',
                        type=str, default="",
                        help='Directory to load predicted masks')
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
                        help="Number of samples processed in one batch.")
    args = parser.parse_args()

    make_submission(args)
