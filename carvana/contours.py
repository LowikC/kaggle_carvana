import numpy as np
import cv2

class_ids = {"background": 0,
             "background_contour": 1,
             "car": 2,
             "car_contour": 3}


def get_contours(mask, thickness=1):
    """
    Get a binary image with the inside and outside contours.
    :param mask: np.array of shape (height, width), np.uint8
    :param thickness: Thickness of the contours.
    :return: np.array of shape (height, width), np.uint8
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_inside_contours_tmp = np.zeros_like(mask, dtype=np.uint8)
    img_outside_contours_tmp = np.zeros_like(mask, dtype=np.uint8)
    # Draw the inside contours.
    cv2.drawContours(img_inside_contours_tmp, contours, -1, (1, 1, 1), thickness)
    img_inside_contours = (img_inside_contours_tmp != 0) & (mask != 0)
    # Draw the inside with thickness = a and outside with thickness = a + 1
    cv2.drawContours(img_outside_contours_tmp, contours, -1, (1, 1, 1), thickness + 1)
    img_outside_contours = (img_outside_contours_tmp != 0) & (mask == 0)

    img_contours = np.zeros_like(img_inside_contours, dtype=np.uint8)
    img_contours[...] = class_ids["background"]
    img_contours[mask != 0] = class_ids["car"]
    img_contours[img_inside_contours] = class_ids["car_contour"]
    img_contours[img_outside_contours] = class_ids["background_contour"]

    return img_contours


def get_pixel_weights(mask, weights_per_class, thickness):
    contours = get_contours(mask, thickness)
    pixel_weights = np.zeros_like(contours, dtype=np.float32)
    for cls in range(weights_per_class.shape[0]):
        pixel_weights[contours == cls] = weights_per_class[cls]
    return pixel_weights


def get_weighted_mask_batch(y_batch, weights_per_class, thickness):
    """
    Get mask and weights for a batch.
    :param y_batch: Np.array (B, H, W, 1).
    :param weights_per_class: np.array (4, ), with the weight of each class.
    :param thickness: Thickness of the contours.
    :return: np.array of size (B, H, W, 2).
    """
    b, h, w, _ = y_batch.shape
    weighted_mask_batch = np.zeros((b, h, w, 2), dtype=np.float32)
    for bid in range(b):
        mask = y_batch[bid, :, :, 0]
        pixel_weights = get_pixel_weights(mask, weights_per_class, thickness)
        weighted_mask_batch[bid, :, :, 0] = mask.astype(np.float32)
        weighted_mask_batch[bid, :, :, 1] = pixel_weights
    return weighted_mask_batch
