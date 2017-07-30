import os
import json
import argparse
import logging
import numpy as np
from FullImageIterator import FullImageIterator
from unet import get_model
from keras.optimizers import SGD, Adam
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import dice_coef_binary, class_weighted_binary_accuracy
from losses import class_weighted_binary_crossentropy, wrapped_partial
from TensorBoardCallBack import TensorBoardCallBack


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor='val_acc', min_delta=0.001,
                      patience=2, mode='max', verbose=1),

        ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                          patience=1, verbose=1, mode='max',
                          epsilon=0.01),

        ModelCheckpoint('.'.join((dst_dir, "weights.{epoch:02d}-{val_acc:.2f}.hdf5")),
                        monitor='val_acc', mode='max', verbose=1)
    ]


def get_data(args):
    with open(args.train, "r") as jfile:
        train_ids = json.load(jfile)

    with open(args.val, "r") as jfile:
        val_ids = json.load(jfile)

    train_generator = FullImageIterator(args.images_dir, args.mask_dir,
                                        train_ids, args.batch_size,
                                        args.image_shape)
    val_generator = FullImageIterator(args.images_dir, args.mask_dir,
                                      val_ids, args.batch_size,
                                      args.image_shape)
    return train_generator, val_generator


def train(args):
    train_generator, val_generator = get_data(args)

    unet = get_model(args.image_shape[0], args.image_shape[1], 3,
                     n_filters=[16, 32, 64, 128, 256])

    weights = np.array([1, 3.75], dtype=np.float32)
    weights /= np.sum(weights)
    weighted_bce_loss = wrapped_partial(class_weighted_binary_crossentropy,
                                        weights=weights)
    weighted_acc = wrapped_partial(class_weighted_binary_accuracy,
                                   weights=weights)

    # opt = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
    opt = Adam()
    unet.compile(optimizer=opt, loss=weighted_bce_loss,
                 metrics=[binary_accuracy, weighted_acc, dice_coef_binary])

    callbacks = get_callbacks(args.save_dir)
    _ = unet.fit_generator(train_generator, 1, #train_generator.steps_per_epoch,
                           epochs=2,
                           verbose=1,
                           validation_data=val_generator,
                           validation_steps=val_generator.steps_per_epoch,
                           callbacks=callbacks,
                           max_queue_size=1,
                           workers=1,
                           initial_epoch=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Train U-Net model')
    parser.add_argument('--save_dir',
                        type=str, default="../data/logs",
                        help='Directory to save checkpoints, logs, ....')
    parser.add_argument('--train',
                        type=str, default="../data/train.json",
                        help='Json file with the training images ids')
    parser.add_argument('--val',
                        type=str, default="../data/val.json",
                        help='Json file with the validation images ids')
    parser.add_argument('--images_dir',
                        type=str, default="../data/train_240x160",
                        help='Path to the directory containing images.')
    parser.add_argument('--mask_dir',
                        type=str, default="../data/train_masks_240x160",
                        help='Path to the directory containing masks.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--image_shape',
                        type=tuple, default=(160, 240),
                        help='(Height, width) of the samples.'
                             'Should match the shape of the images'
                             ' (will be resized otherwise).')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)
