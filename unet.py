import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
import h5py
from segmentation_models.metrics import dice_score, jaccard_score
from keras.metrics import binary_crossentropy
from keras.layers import Conv2D, Input, MaxPooling2D, concatenate, UpSampling2D, PReLU, BatchNormalization, ReLU, add
from keras.models import Model

SEED = 42
smooth = 1e-10
HEIGHT, WIDTH, DEPTH = 224, 224, 1
IMAGES = 'E:/datasets/parking/images'
MASKS = 'E:/datasets/parking/masks'
BATCH = 4


def residual(in_filters, out_filters, kernel_size=(3, 3), bottleneck_rate=4, dilation=(1, 1)):
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

        x = add([x, a])
        return x

    return layer


def conv(x, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def Unet():
    input_layer = Input(shape=(HEIGHT, WIDTH, 3))

    conv0 = conv(input_layer, 16)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = conv(pool0, 16)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv(pool1, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv(pool3, 128)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv(pool4, 256)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = conv(up6, 128)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = conv(up7, 64)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = conv(up8, 32)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = conv(up9, 16)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv0], axis=3)
    conv10 = conv(up10, 16)

    conv11 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv10)

    model = Model(input=input_layer, output=conv11)

    return model


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        rotation_range=45,
        vertical_flip=True,
        horizontal_flip=True,
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        rotation_range=45,
        vertical_flip=True,
        horizontal_flip=True,
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def prepare_data():
    print('starting making data..')

    dataset_name = 'birdEyeView.hdf5'

    if os.path.isfile(dataset_name):
        data = h5py.File(dataset_name, 'r')
        print('read dataset from hdf5')
        return data['images'][()], data['masks'][()]

    images = os.listdir(IMAGES)
    masks = os.listdir(MASKS)

    x_data = np.empty((len(images), HEIGHT, WIDTH, 3), dtype=np.uint8)
    y_data = np.empty((len(masks), HEIGHT, WIDTH, 1), dtype=np.uint8)

    tbar = tqdm(images)
    for i, file_name in enumerate(tbar):
        image = cv2.imread(os.path.join(IMAGES, file_name), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(os.path.join(MASKS, file_name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)
        mask[mask != 255] = 0
        mask = mask[:, :, np.newaxis]

        x_data[i] = image
        y_data[i] = mask

    print(f'{len(x_data)} images loaded!')

    data = h5py.File(dataset_name, 'w')

    data.create_dataset('images', data=x_data)
    data.create_dataset('masks', data=y_data)

    data.close()

    return x_data, y_data


def loss(y_true, y_pred):
    return 1.0 * binary_crossentropy(y_true, y_pred) + 1.0 * (1. - jaccard_score(y_true, y_pred))


if __name__ == '__main__':
    x_data, y_data = prepare_data()

    train_images, val_images, train_masks, val_masks = x_data[:5000], x_data[5000:], y_data[:5000], y_data[5000:]

    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(train_images[10])
    # axes[0].set_title('image')
    # axes[1].imshow(train_masks[10][:, :, 0])
    # axes[1].set_title('mask')
    # plt.show()
    #
    # exit()
    callbacks_list = [
        ModelCheckpoint('models/unet_rgb' + str(BATCH) + '_batch.h5',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True),
        TensorBoard(log_dir='./logs',
                    batch_size=BATCH,
                    write_images=True),
        ReduceLROnPlateau(verbose=1, factor=0.25, patience=3, min_lr=1e-6)
    ]

    model = Unet()

    model.summary()
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=[dice_score, jaccard_score])

    model_json = model.to_json()
    json_file = open('models/unet_rgb' + str(BATCH) + '_batch.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    model.fit_generator(
        my_generator(train_images, train_masks, BATCH),
        steps_per_epoch=len(train_masks) / BATCH,
        epochs=50,
        verbose=1,
        validation_data=my_generator(val_images, val_masks, 1),
        validation_steps=len(val_images),
        callbacks=callbacks_list,
        shuffle=True
    )

    print('done!')
