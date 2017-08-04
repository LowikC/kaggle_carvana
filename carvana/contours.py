import numpy as np
import cv2

class_ids = {"background": 0,
             "background_contour": 1,
             "car": 2,
             "car_contour": 3}


def get_contours(mask):
    """
    Get a binary image with the inside and outside contours.
    :param mask: np.array of shape (height, width), np.uint8
    :return: np.array of shape (height, width), np.uint8
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_inside_contours = np.zeros_like(mask, dtype=np.uint8)
    img_outside_contours_tmp = np.zeros_like(mask, dtype=np.uint8)
    # Draw the inside contours.
    cv2.drawContours(img_inside_contours, contours, -1, (1, 1, 1), 1)
    # Draw the outside with thickness = 1 and inside with thickness = 2
    cv2.drawContours(img_outside_contours_tmp, contours, -1, (1, 1, 1), 2)
    img_outside_contours = (img_outside_contours_tmp != 0) & (mask == 0)

    img_contours = np.zeros_like(img_inside_contours, dtype=np.uint8)
    img_contours[...] = class_ids["background"]
    img_contours[mask != 0] = class_ids["car"]
    img_contours[img_inside_contours != 0] = class_ids["car_contour"]
    img_contours[img_outside_contours] = class_ids["background_contour"]

    return img_contours
