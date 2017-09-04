from metrics import weighted_dice_coef
from keras import backend as K
from functools import partial, update_wrapper


# This is needed to keep the __name__ attribute
# of the partial func (needed in Keras)
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def weighted_dice_coef_loss(y_true, y_pred):
    """
    Differentiable (weighted) dice coef loss.
    """
    return 1.0 - weighted_dice_coef(y_true, y_pred, 1.0)


def weighted_binary_crossentropy(y_true, y_pred):
    """
    (weighted) binary cross entropy.
    """
    y_true_mask = y_true[..., 0:1]
    y_true_weights = y_true[..., 1:2]
    loss_per_pixel = K.binary_crossentropy(y_pred, y_true_mask)
    return K.sum(loss_per_pixel * y_true_weights)/K.sum(y_true_weights)
