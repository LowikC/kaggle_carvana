from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D


def unet_down_block(x, n_filters, block_id, with_maxpool=True, activation="elu",
                    crop=False):
    padding = 'valid' if crop else 'same'
    y = Conv2D(n_filters, (3, 3), activation=activation,
               padding=padding, name="conv{}_1".format(block_id))(x)
    y = Conv2D(n_filters, (3, 3), activation=activation,
               padding=padding, name="conv{}_2".format(block_id))(y)
    if not with_maxpool:
        return y

    pool = MaxPooling2D(pool_size=(2, 2), name="max_pool{}".format(block_id))(y)
    return y, pool


def unet_up_block(x, y, n_filters, block_id, activation="elu", crop=False):
    padding = 'valid' if crop else 'same'
    up_x = UpSampling2D(size=(2, 2), name="upsample{}".format(block_id))(x)

    # Compute crop needed to have the same shape for up_x and y
    if crop:
        _, hx, wx, _ = up_x.shape
        _, hy, wy, _ = y.shape
        cropy = int(hy - hx) // 2
        cropx = int(wy - wx) // 2
        crop_y = Cropping2D(cropping=((cropy, cropy), (cropx, cropx)),
                            name="crop{}".format(block_id))(y)
    else:
        crop_y = y
    up = concatenate([up_x, crop_y], axis=-1,
                     name="concat{}".format(block_id))
    up = Conv2D(n_filters, (3, 3),
                activation=activation,
                padding=padding,
                name="conv{}_1".format(block_id))(up)
    up = Conv2D(n_filters, (3, 3),
                activation=activation,
                padding=padding,
                name="conv{}_2".format(block_id))(up)
    return up


def get_model(im_height, im_width, n_channels=3,
              n_filters=(64, 128, 256, 512, 1024)):
    inputs = Input((im_height, im_width, n_channels))

    conv1, pool1 = unet_down_block(inputs, n_filters[0], 1)
    conv2, pool2 = unet_down_block(pool1, n_filters[1], 2)
    conv3, pool3 = unet_down_block(pool2, n_filters[2], 3)
    conv4, pool4 = unet_down_block(pool3, n_filters[3], 4)
    conv5 = unet_down_block(pool4, n_filters[4], 5, with_maxpool=False)

    conv6 = unet_up_block(conv5, conv4, n_filters[3], 6)
    conv7 = unet_up_block(conv6, conv3, n_filters[2], 7)
    conv8 = unet_up_block(conv7, conv2, n_filters[1], 8)
    conv9 = unet_up_block(conv8, conv1, n_filters[0], 9)

    segmentation = Conv2D(1, (1, 1), activation='sigmoid', name="segmentation")(
        conv9)

    model = Model(inputs=[inputs], outputs=[segmentation], name="unet")

    return model


def preprocess(x):
    return (x - 127) / 255
