import os
import json
import argparse
import logging
import numpy as np
from ImageMaskIterator import ImageMaskIterator
from tiramisu import DenseNetFCN, preprocess
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import dice_coef_binary
from losses import dice_coef_loss
from TensorBoardCallBack import TensorBoardCallBack


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    # monitor = 'val_dice_coef_binary_contours'
    monitor = 'val_loss'
    ckpt_name = "weights.{epoch:02d}-{val_dice_coef_binary:.4f}-{" + monitor + "}.hdf5"

    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor=monitor, min_delta=0.0001,
                      patience=2, mode='min', verbose=1),

        ReduceLROnPlateau(monitor=monitor, factor=0.1,
                          patience=1, verbose=1, mode='min',
                          epsilon=0.005),

        ModelCheckpoint('.'.join((dst_dir, ckpt_name)),
                        monitor=monitor, mode='min', verbose=1)
    ]


def get_data(args):
    with open(args.train, "r") as jfile:
        train_ids = json.load(jfile)

    with open(args.val, "r") as jfile:
        val_ids = json.load(jfile)

    image_shape = (args.image_height, args.image_width)
    train_generator = ImageMaskIterator(args.images_dir, args.masks_dir,
                                        train_ids, args.batch_size,
                                        image_shape,
                                        x_preprocess=preprocess)
    val_generator = ImageMaskIterator(args.images_dir, args.masks_dir,
                                      val_ids, args.batch_size,
                                      image_shape,
                                      x_preprocess=preprocess)
    return train_generator, val_generator


def train(args):
    train_generator, val_generator = get_data(args)

    input_shape = (args.image_height, args.image_width, 3)
    tiramisu = DenseNetFCN(input_shape, 5, 12, 4, activation='sigmoid',
                           init_conv_filters=24)

    opt = Adam()
    tiramisu.compile(optimizer=opt, loss=dice_coef_loss,
                     metrics=[binary_accuracy, dice_coef_binary])

    callbacks = get_callbacks(args.save_dir)
    _ = tiramisu.fit_generator(train_generator, train_generator.steps_per_epoch,
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
        description='Train Tiramisu model')
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
