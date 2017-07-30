from metrics import dice_coef
from keras import backend as K
from functools import partial, update_wrapper


# This is needed to keep the __name__ attribute of the partial func (needed in Keras)
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred, 1.0)


def log_dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred, 1.0))


def class_weighted_binary_crossentropy(y_true, y_pred, weights):
    loss_per_pixel = K.binary_crossentropy(y_pred, y_true)
    weights_per_pixel = y_true * weights[1] + (1 - y_true) * weights[0]
    return loss_per_pixel * weights_per_pixel


def pixel_weighted_binary_crossentropy(y_true, y_pred):
    y_true_weights = y_true[..., 1:2]
    y_true_mask = y_true[..., 0:1]
    loss_per_pixel = K.binary_crossentropy(y_pred, y_true_mask)
    return loss_per_pixel * y_true_weights
