import os
import json
import argparse
import logging
import numpy as np
from ImageMaskIterator import ImageMaskIterator
from unet import get_model, preprocess
from optimizers import AdamWithAcc
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import dice_coef_binary
from losses import weighted_binary_crossentropy, weighted_dice_coef_loss, wrapped_partial
from TensorBoardCallBack import TensorBoardCallBack
from contours import get_weighted_mask_batch


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    # monitor = 'val_dice_coef_binary_contours'
    monitor = 'val_loss'
    ckpt_name = "weights.{epoch:02d}-{val_dice_coef_binary_contours:.4f}-{" + monitor + "}.hdf5"
    mode = 'min'
    epsilon = 0.005
    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor=monitor, min_delta=epsilon/10,
                      patience=2, mode=mode, verbose=1),

        ReduceLROnPlateau(monitor=monitor, factor=0.1,
                          patience=1, verbose=1, mode=mode,
                          epsilon=epsilon),

        ModelCheckpoint('.'.join((dst_dir, ckpt_name)),
                        monitor=monitor, mode=mode, verbose=1)
    ]


def get_data(args):
    with open(args.train, "r") as jfile:
        train_ids = json.load(jfile)

    with open(args.val, "r") as jfile:
        val_ids = json.load(jfile)

    image_shape = (args.image_height, args.image_width)
    weights_per_class = np.array([1.0, 1.0, 1.0, 1.0])
    thickness = 1
    get_weighted_mask_batch_s = wrapped_partial(get_weighted_mask_batch,
                                                weights_per_class=weights_per_class,
                                                thickness=thickness)
    train_generator = ImageMaskIterator(images_dir=args.images_dir,
                                        images_ids=train_ids,
                                        images_ext=args.images_ext,
                                        masks_dir=args.masks_dir,
                                        masks_ext=args.masks_ext,
                                        batch_size=args.batch_size,
                                        x_shape=image_shape,
                                        x_preprocess=preprocess,
                                        y_preprocess=get_weighted_mask_batch_s)
    val_generator = ImageMaskIterator(images_dir=args.images_dir,
                                      images_ids=val_ids,
                                      images_ext=args.images_ext,
                                      masks_dir=args.masks_dir,
                                      masks_ext=args.masks_ext,
                                      batch_size=args.batch_size,
                                      x_shape=image_shape,
                                      x_preprocess=preprocess,
                                      y_preprocess=get_weighted_mask_batch_s)
    return train_generator, val_generator


def bce_dice_loss(y_true, y_pred, bce_loss, dice_loss,
                  weights_losses):
    return weights_losses[0] * bce_loss(y_true, y_pred) + \
           weights_losses[1] * dice_loss(y_true, y_pred)


def train(args):
    train_generator, val_generator = get_data(args)

    unet = get_model(args.image_height, args.image_width, 3,
                     n_filters=[16, 32, 64, 128, 256])

    loss = wrapped_partial(bce_dice_loss,
                           bce_loss=weighted_binary_crossentropy,
                           dice_loss=weighted_dice_coef_loss,
                           weights_losses=[0.5, 0.5])

    for bx, by in train_generator:
        break

    print("hihi")
    opt = Adam() #AdamWithAcc(accum_iters=args.batch_accum)
    unet.compile(optimizer=opt, loss=loss,
                 metrics=[weighted_binary_crossentropy, weighted_dice_coef_loss,
                          dice_coef_binary, binary_accuracy])
    print("jjj")
    ypred = unet.predict(bx)
    print("hihi2")

    g = loss(by, ypred)

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
                        type=str, default="data/train",
                        help='Path to the directory containing images.')
    parser.add_argument('--images_ext',
                        type=str, default=".png",
                        help='Extension of images files.')
    parser.add_argument('--masks_dir',
                        type=str, default="data/train_masks",
                        help='Path to the directory containing masks.')
    parser.add_argument('--masks_ext',
                        type=str, default=".png",
                        help='Extension of masks files.')
    parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='Batch size.')
    parser.add_argument('--batch_accum',
                        type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--image_width',
                        type=int, default=1920,
                        help='Width of the samples.')
    parser.add_argument('--image_height',
                        type=int, default=1280,
                        help='Height of the samples.')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)
