import os
import json
import argparse
import logging
import numpy as np
from ImageMaskIterator import ImageMaskIterator
from unet import get_model, preprocess
from AdamAccumulate import Adam_accumulate
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import dice_coef_binary
from losses import background_weighted_binary_crossentropy, dice_coef_loss, wrapped_partial
from augmentation import random_transformation
from TensorBoardCallBack import TensorBoardCallBack


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    # monitor = 'val_dice_coef_binary_contours'
    monitor = 'val_loss'
    mode = 'min'
    ckpt_name = "weights.{epoch:02d}-{val_dice_coef_binary:.4f}-{" + monitor + ":.4f}.hdf5"

    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor=monitor, min_delta=0.0001,
                      patience=2, mode=mode, verbose=1),

        ReduceLROnPlateau(monitor=monitor, factor=0.1,
                          patience=1, verbose=1, mode=mode,
                          epsilon=0.005),

        ModelCheckpoint('.'.join((dst_dir, ckpt_name)),
                        monitor=monitor, mode=mode, verbose=1)
    ]


def get_data(args):
    with open(args.train, "r") as jfile:
        train_ids = json.load(jfile)

    with open(args.val, "r") as jfile:
        val_ids = json.load(jfile)

    x_shape = (args.image_height, args.image_width)

    train_generator = ImageMaskIterator(images_dir=args.images_dir,
                                        images_ids=train_ids,
                                        images_ext=args.images_ext,
                                        masks_dir=args.masks_dir,
                                        masks_ext=args.masks_ext,
                                        batch_size=args.batch_size,
                                        n_patches_per_image=args.patches_per_image,
                                        x_shape=x_shape,
                                        x_preprocess=preprocess,
                                        data_augmentation=random_transformation)

    val_generator = ImageMaskIterator(images_dir=args.images_dir,
                                      images_ids=val_ids,
                                      images_ext=args.images_ext,
                                      masks_dir=args.masks_dir,
                                      masks_ext=args.masks_ext,
                                      batch_size=args.batch_size,
                                      x_shape=x_shape,
                                      x_preprocess=preprocess)
    return train_generator, val_generator


def bce_dice_loss(y_true, y_pred, bce_loss, dice_loss, weights):
    return weights[0] * bce_loss(y_true, y_pred) + weights[1] * dice_loss(y_true, y_pred)


def train(args):
    train_generator, val_generator = get_data(args)

    unet = get_model(args.image_height, args.image_width, 3,
                     n_filters=[24, 48, 96, 192, 384])

    weights = np.array([1, 1.54], dtype=np.float32)  # 60% of background
    weights /= np.sum(weights)
    weighted_bce_loss = wrapped_partial(background_weighted_binary_crossentropy,
                                        weights=weights)
    bce_dice_loss_spec = wrapped_partial(bce_dice_loss,
                                         bce_loss=weighted_bce_loss,
                                         dice_loss=dice_coef_loss,
                                         weights=[0.5, 0.5])

    opt = Adam_accumulate(accum_iters=4)
    unet.compile(optimizer=opt, loss=bce_dice_loss_spec,
                 metrics=[binary_accuracy, dice_coef_binary])

    callbacks = get_callbacks(args.save_dir)
    _ = unet.fit_generator(train_generator, train_generator.steps_per_epoch,
                           epochs=100,
                           verbose=1,
                           validation_data=val_generator,
                           validation_steps=val_generator.steps_per_epoch,
                           callbacks=callbacks,
                           max_queue_size=8,
                           workers=4,
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
                        type=str, default="data/train",
                        help='Path to the directory containing images.')
    parser.add_argument('--images_ext',
                        type=str, default=".jpg",
                        help='Extension of images files.')
    parser.add_argument('--masks_dir',
                        type=str, default="data/train_masks",
                        help='Path to the directory containing masks.')
    parser.add_argument('--masks_ext',
                        type=str, default=".gif",
                        help='Extension of masks files.')
    parser.add_argument('--batch_size',
                        type=int, default=8,
                        help='Batch size.')
    parser.add_argument('--patches_per_image',
                        type=int, default=4,
                        help='Number of patches taken from the same image in a batch.')
    parser.add_argument('--image_width',
                        type=int, default=512,
                        help='Width of the samples.')
    parser.add_argument('--image_height',
                        type=int, default=512,
                        help='Height of the samples.')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)
