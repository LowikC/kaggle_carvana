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


def background_weighted_binary_crossentropy(y_true, y_pred, weights):
    """
    Compute a weighted binary cross entropy, by weighting the background and
        foreground,
    :param y_true: 
    :param y_pred: 
    :param weights: 
    :return: 
    """
    loss_per_pixel = K.binary_crossentropy(y_pred, y_true)
    weights_per_pixel = y_true * weights[1] + (1 - y_true) * weights[0]
    return loss_per_pixel * weights_per_pixel


def contours_weighted_binary_crossentropy(y_true, y_pred, weights):
    """
    Compute a weighted binary cross entropy, by weighting the different 
        type of mask pixels (contours).
    :param y_true: A Tensor (b, h, w, 1), with the class of each pixel.
        The classes must be:
            - 0: background, 
            - 1: background_contour,
            - 2: foreground,
            - 3: foreground contour.
    :param y_pred: A Tensor (b, h, w, 1) with the mask prediction, in [0, 1]
    :param weights: A np.array of shape (4,), containing the weight of each class.
        It should be scaled so that the total loss stay close to the non-weighted one.
    :return: The weighted cross-entropy.
    """
    # The binary classification is between classes (0, 1) and (2, 3)
    y_mask = K.cast(y_true > 1, K.floatx())
    loss_per_pixel = K.binary_crossentropy(y_pred, y_mask)

    # Compute the weight of each pixel, based on its class.
    background = K.cast(K.equal(y_true, 0), K.floatx())
    background_contours = K.cast(K.equal(y_true, 1), K.floatx())
    foreground = K.cast(K.equal(y_true, 2), K.floatx())
    foreground_contours = K.cast(K.equal(y_true, 3), K.floatx())

    weights_per_pixel = background * weights[0] + \
                        background_contours * weights[1] + \
                        foreground * weights[2] + \
                        foreground_contours * weights[3]

    return loss_per_pixel * weights_per_pixel


def pixel_weighted_binary_crossentropy(y_true, y_pred):
    y_true_weights = y_true[..., 1:2]
    y_true_mask = y_true[..., 0:1]
    loss_per_pixel = K.binary_crossentropy(y_pred, y_true_mask)
    return loss_per_pixel * y_true_weights
