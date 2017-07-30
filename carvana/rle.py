import numpy as np


def encode(mask):
    pixels = mask.ravel()
    pixels_left = np.hstack((np.array([False]), pixels))
    pixels_right = np.hstack((pixels, np.array([False])))
    runs = np.where(pixels_left != pixels_right)[0]
    return runs[::2] + 1, runs[1::2] - runs[::2]


def dumps(mask):
    starts, lengths = encode(mask)
    return " ".join((str(s) + " " + str(l)
                     for s, l in zip(starts, lengths)))


def loads(rle, shape=(1280, 1918)):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    tokens = rle.split()
    assert(len(tokens) % 2 == 0)
    for i in range(0, len(tokens), 2):
        start = int(tokens[i]) - 1  # rle is indexed from 1
        length = int(tokens[i+1])
        mask[start:start+length] = 1
    return mask.reshape(shape)
