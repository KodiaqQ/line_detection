import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, PReLU, concatenate, SpatialDropout2D, BatchNormalization, add, \
    UpSampling2D, Conv2DTranspose, Lambda, Permute, ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.backend as K


def initial(filters=3, kernel_size=(3, 3)):
    def layer(x):
        a = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)

        b = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = concatenate([a, b], axis=-1)
        return x

    return layer


def bottleneck(in_filters, out_filters, kernel_size=(3, 3), bottleneck_rate=4, dilation=(1, 1), dropout_rate=0.1):
    def layer(x):
        a = Conv2D(filters=in_filters // bottleneck_rate, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=in_filters // bottleneck_rate, kernel_size=kernel_size, dilation_rate=dilation,
                   padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=out_filters, kernel_size=(1, 1), padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = SpatialDropout2D(rate=dropout_rate)(a)

        x = add([x, a])
        return x

    return layer


def bottleneck_assymetric(in_filters, out_filters, kernel_size=5, bottleneck_rate=4, dropout_rate=0.1):
    def layer(x):
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=(kernel_size, 1), padding='same', use_bias=False)(
            a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=(1, kernel_size), padding='same', use_bias=False)(
            a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=out_filters, kernel_size=(1, 1), padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = SpatialDropout2D(rate=dropout_rate)(a)

        x = add([x, a])
        return x

    return layer


def bottleneck_pool(in_filters, out_filters, kernel_size=(3, 3), bottleneck_rate=4, dropout_rate=0.1):
    def layer(x):
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=(2, 2), strides=(2, 2), padding='same',
                   use_bias=False)(x)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=kernel_size, padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=out_filters, kernel_size=(1, 1), padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = SpatialDropout2D(rate=dropout_rate)(a)

        b = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        b = Permute((1, 3, 2))(b)
        pad_feature_maps = a.shape[0], a.shape[1], a.shape[2], a.shape[3] - b.shape[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        b = ZeroPadding2D(padding=(tb_pad, lr_pad))(b)
        b = Permute((1, 3, 2))(b)

        # b = concatenate([b, K.zeros(a.shape[0], a.shape[1], a.shape[2], a.shape[3] - b.shape[3])])

        x = add([b, a])
        return x

    return layer


def bottleneck_unpool(in_filters, out_filters, kernel_size=(3, 3), bottleneck_rate=4, dropout_rate=0.1):
    def layer(x):
        a = Conv2D(filters=in_filters / bottleneck_rate, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2DTranspose(filters=in_filters / bottleneck_rate, kernel_size=kernel_size, strides=(2, 2),
                            padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)
        a = Conv2D(filters=out_filters, kernel_size=(1, 1), padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = PReLU(shared_axes=[1, 2])(a)

        b = Conv2D(filters=out_filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        b = BatchNormalization()(b)
        b = PReLU(shared_axes=[1, 2])(b)
        b = UpSampling2D(size=(2, 2))(b)

        x = add([b, a])

        return x

    return layer


def ENet(image_shape, n_classes):
    input_layer = Input(shape=image_shape)

    init = initial()(input_layer)

    bn_10 = bottleneck_pool(16, 64, bottleneck_rate=1, dropout_rate=0.1)(init)
    bn_11 = bottleneck(64, 64, dropout_rate=0.01)(bn_10)
    bn_12 = bottleneck(64, 64, dropout_rate=0.01)(bn_11)
    bn_13 = bottleneck(64, 64, dropout_rate=0.01)(bn_12)
    bn_14 = bottleneck(64, 64, dropout_rate=0.01)(bn_13)

    bn_20 = bottleneck_pool(64, 128)(bn_14)
    bn_21 = bottleneck(128, 128)(bn_20)
    bn_22 = bottleneck(128, 128, dilation=(2, 2))(bn_21)
    bn_23 = bottleneck_assymetric(128, 128)(bn_22)
    bn_24 = bottleneck(128, 128, dilation=(4, 4))(bn_23)
    bn_25 = bottleneck(128, 128)(bn_24)
    bn_26 = bottleneck(128, 128, dilation=(8, 8))(bn_25)
    bn_27 = bottleneck_assymetric(128, 128)(bn_26)
    bn_28 = bottleneck(128, 128, dilation=(16, 16))(bn_27)

    bn_31 = bottleneck(128, 128)(bn_28)
    bn_32 = bottleneck(128, 128, dilation=(2, 2))(bn_31)
    bn_33 = bottleneck_assymetric(128, 128)(bn_32)
    bn_34 = bottleneck(128, 128, dilation=(4, 4))(bn_33)
    bn_35 = bottleneck(128, 128)(bn_34)
    bn_36 = bottleneck(128, 128, dilation=(8, 8))(bn_35)
    bn_37 = bottleneck_assymetric(128, 128)(bn_36)
    bn_38 = bottleneck(128, 128, dilation=(16, 16))(bn_37)

    bn_40 = bottleneck_unpool(128, 64)(bn_38)
    bn_41 = bottleneck(64, 64)(bn_40)
    bn_42 = bottleneck(64, 64)(bn_41)

    bn_50 = bottleneck_unpool(64, 16)(bn_42)
    bn_51 = bottleneck(16, 16, bottleneck_rate=2)(bn_50)
    output_layer = Conv2DTranspose(filters=n_classes, kernel_size=(1, 1))(bn_51)

    model = Model(input_layer, output_layer)
    return model


if __name__ == '__main__':
    model = ENet(image_shape=(224, 224, 3), n_classes=3)
