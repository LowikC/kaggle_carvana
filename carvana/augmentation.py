import cv2
import numpy as np


def flip(x, y, apply_chance=0.5):
    if np.random.rand(1) < apply_chance:
        x[...] = np.flip(x, 1)
        y[...] = np.flip(y, 1)


def hue_change(x, y, r=(-180, 180)):
    hue_diff = np.random.randint(r[0], r[1])
    x[..., 0] = np.maximum(0, np.minimum(179, x[..., 0].astype(np.int32) - hue_diff))


def sat_change(x, y, r=(0, 255)):
    sat_diff = np.random.randint(r[0], r[1])
    x[..., 1] = np.maximum(0, np.minimum(255, x[..., 1].astype(np.int32) - sat_diff))


def val_change(x, y, r=(-30, 30)):
    val_diff = np.random.randint(r[0], r[1])
    x[..., 2] = np.maximum(0, np.minimum(255, x[..., 2].astype(np.int32) - val_diff))


def zoom(x, y, apply_chance=0.3, zmax=1.2, zdiffmax=0.1):
    apply_zoom = np.random.rand(1)
    if apply_zoom > apply_chance:
        return

    h, w, c = x.shape
    zy = 1 + np.random.rand(1) * (zmax - 1)
    zx = 1 + np.random.rand(1) * (zmax - 1)
    if zy > zx:
        zy = min(zy, zx + zdiffmax)
    else:
        zx = min(zx, zy + zdiffmax)

    hz = int(np.round(h/zy))
    wz = int(np.round(w/zx))
    dh = (h - hz) // 2
    dw = (w - wz) // 2
    xz = x[dh: dh + hz, dw: dw + wz, ...]
    yz = y[dh: dh + hz, dw: dw + wz, ...]
    x[...] = cv2.resize(xz, (w, h), interpolation=cv2.INTER_CUBIC)
    maskz = cv2.resize(yz * 255, (w, h), interpolation=cv2.INTER_CUBIC)
    y[...] = (maskz > 127).astype(np.uint8)


def random_transformation(x, y):
    y_cpy = y.copy()
    x_hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)

    hue_change(x_hsv, y_cpy)
    sat_change(x_hsv, y_cpy)
    val_change(x_hsv, y_cpy)
    flip(x_hsv, y_cpy)
    x_rgb = cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB)
    zoom(x_rgb, y_cpy)

    return x_rgb, y_cpy
