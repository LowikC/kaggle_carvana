from keras import backend as K


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) /\
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_binary(y_true, y_pred):
    y_pred_binary = K.round(y_pred)
    return dice_coef(y_true, y_pred_binary, K.epsilon())


def dice_coef_binary_contours(y_true, y_pred):
    y_true_binary = K.cast(y_true > 1, K.floatx())
    y_pred_binary = K.round(y_pred)
    return dice_coef(y_true_binary, y_pred_binary, K.epsilon())


def background_weighted_binary_accuracy(y_true, y_pred, weights):
    weight_per_pixel = y_true * weights[1] + (1 - y_true) * weights[0]
    accuracy_per_pixel = K.equal(y_true, K.round(y_pred))
    accuracy_per_pixel = K.cast(accuracy_per_pixel, K.floatx())
    return K.mean(accuracy_per_pixel * weight_per_pixel, axis=-1)


def pixel_weighted_binary_accuracy(y_true, y_pred):
    y_true_mask = y_true[..., 0:1]
    return K.mean(K.equal(y_true_mask, K.round(y_pred)), axis=-1)

