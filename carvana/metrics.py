from keras import backend as K


def weighted_dice_coef(y_true, y_pred, smooth, use_weights=True):
    """
    Compute differentiable (weighted) dice coef.
    :param y_true: Groundtruth mask + weight per pixel (optional)
    :param y_pred: Predicted mask
    :param smooth: Smooth factor
    :param use_weights: Use the weights or not.
    :return: weighted dice coef.
    """
    y_true_weights = K.ones_like(y_true)
    y_true_mask = y_true[..., 0:1]
    if use_weights:
        y_true_weights = y_true[..., 1:2]
    y_true_mask_f = K.flatten(y_true_mask)
    y_true_weights_f = K.flatten(y_true_weights)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_mask * y_pred_f)
    return (2. * y_true_weights_f * intersection + smooth) / \
           (K.sum(y_true_weights_f * y_true_mask_f) +
            K.sum(y_true_weights_f * y_pred_f) + smooth)


def dice_coef_binary(y_true, y_pred):
    """
    Compute the dice coef (on binary prediction)
    """
    y_pred_binary = K.round(y_pred)
    return weighted_dice_coef(y_true, y_pred_binary, K.epsilon(), use_weights=False)


def binary_accuracy(y_true, y_pred):
    if int(y_true.shape[-1]) == 2:
        y_true_mask = y_true[..., 0:1]
    else:
        y_true_mask = y_true
    return K.mean(K.equal(y_true_mask, K.round(y_pred)), axis=-1)
