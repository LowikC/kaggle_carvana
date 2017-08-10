import os
import json
import argparse
import logging
import numpy as np
from FullImageWithContoursIterator import FullImageWithContoursIterator
from unet import get_model, preprocess
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import dice_coef_binary_contours
from losses import contours_weighted_binary_crossentropy, wrapped_partial
from TensorBoardCallBack import TensorBoardCallBack
from contours import get_contours_batch


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor='val_dice_coef_binary_contours', min_delta=0.0001,
                      patience=2, mode='max', verbose=1),

        ReduceLROnPlateau(monitor='val_dice_coef_binary_contours', factor=0.1,
                          patience=1, verbose=1, mode='max',
                          epsilon=0.005),

        ModelCheckpoint('.'.join((dst_dir, "weights.{epoch:02d}-{val_dice_coef_binary_contours:.4f}.hdf5")),
                        monitor='val_dice_coef_binary_contours', mode='max', verbose=1)
    ]


def get_data(args):
    with open(args.train, "r") as jfile:
        train_ids = json.load(jfile)

    with open(args.val, "r") as jfile:
        val_ids = json.load(jfile)

    image_shape = (args.image_height, args.image_width)
    train_generator = FullImageWithContoursIterator(args.images_dir, args.masks_dir,
                                                    train_ids, args.batch_size,
                                                    image_shape,
                                                    xpreprocess=preprocess,
                                                    ypreprocess=get_contours_batch)
    val_generator = FullImageWithContoursIterator(args.images_dir, args.masks_dir,
                                                  val_ids, args.batch_size,
                                                  image_shape,
                                                  xpreprocess=preprocess,
                                                  ypreprocess=get_contours_batch)
    return train_generator, val_generator


def train(args):
    train_generator, val_generator = get_data(args)

    unet = get_model(args.image_height, args.image_width, 3,
                     n_filters=[16, 32, 64, 128, 256])

    weights = np.array([0.21228926, 99.7778963, 0.79766402, 127.05663031], dtype=np.float32)
    weighted_bce_loss = wrapped_partial(contours_weighted_binary_crossentropy,
                                        weights=weights)

    opt = Adam()
    unet.compile(optimizer=opt, loss=weighted_bce_loss,
                 metrics=[binary_accuracy, dice_coef_binary_contours])

    callbacks = get_callbacks(args.save_dir)
    _ = unet.fit_generator(train_generator, train_generator.steps_per_epoch,
                           epochs=100,
                           verbose=1,
                           validation_data=val_generator,
                           validation_steps=val_generator.steps_per_epoch,
                           callbacks=callbacks,
                           max_queue_size=1,
                           workers=1,
                           initial_epoch=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    parser = argparse.ArgumentParser(
        description='Train U-Net model')
    parser.add_argument('--save_dir',
                        type=str, default="data/logs_960x640",
                        help='Directory to save checkpoints, logs, ....')
    parser.add_argument('--train',
                        type=str, default="data/train.json",
                        help='Json file with the training images ids')
    parser.add_argument('--val',
                        type=str, default="data/val.json",
                        help='Json file with the validation images ids')
    parser.add_argument('--images_dir',
                        type=str, default="data/train_960x640",
                        help='Path to the directory containing images.')
    parser.add_argument('--masks_dir',
                        type=str, default="data/train_masks_960x640",
                        help='Path to the directory containing masks.')
    parser.add_argument('--batch_size',
                        type=int, default=4,
                        help='Batch size.')
    parser.add_argument('--image_width',
                        type=int, default=960,
                        help='Width of the samples.'
                             'Should match the shape of the images'
                             ' (will be resized otherwise).')
    parser.add_argument('--image_height',
                        type=int, default=640,
                        help='Height of the samples.'
                             'Should match the shape of the images'
                             ' (will be resized otherwise).')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)
