from segmentation_models import Linknet
from segmentation_models.utils import set_trainable
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adadelta
import h5py
from segmentation_models.metrics import dice_score, jaccard_score
from keras.metrics import binary_crossentropy
from albumentations import *

SEED = 42
smooth = 1e-10
HEIGHT, WIDTH, DEPTH = 224, 224, 1
IMAGES = 'E:/datasets/parking/images'
MASKS = 'E:/datasets/parking/masks'
BATCH = 4


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator().flow(x_train, x_train, batch_size, seed=SEED, shuffle=True)
    mask_generator = ImageDataGenerator().flow(y_train, y_train, batch_size, seed=SEED, shuffle=True)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        X = np.empty((batch_size, x_batch[0].shape[0], x_batch[0].shape[1], x_batch[0].shape[2]), dtype='float32')
        y = np.empty((batch_size, x_batch[0].shape[0], x_batch[0].shape[1], x_batch[0].shape[2]), dtype='float32')

        for i, image in enumerate(x_batch):
            image = np.array(image, dtype=np.uint8)

            sample = {'image': image, 'mask': y_batch[0, :, :, :]}
            augmentation = aug()
            augmentations = augmentation(**sample)

            X[i], y[i] = augmentations['image'] / 255., augmentations['mask'] / 255.

        yield X, y


def prepare_data():
    print('starting making data..')

    dataset_name = 'birdEyeView_binary.hdf5'

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
        image = cv2.GaussianBlur(image, (5, 5), 0)
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


def aug(p=1):
    return Compose([
        OneOf([
            HorizontalFlip(),
            VerticalFlip(),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=90),
            RandomRotate90()
        ], p=0.75),
        JpegCompression(p=0.25),
        CLAHE(p=0.25),
        MedianBlur(p=0.25),
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.25)
    ], p=p)


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
        ModelCheckpoint('models/linknet_gray' + str(BATCH) + '_batch.h5',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True),
        TensorBoard(log_dir='./logs',
                    batch_size=BATCH,
                    write_images=True),
        ReduceLROnPlateau(verbose=1, factor=0.25, patience=3, min_lr=1e-6)
    ]

    model = Linknet(
        backbone_name='mobilenetv2',
        input_shape=(HEIGHT, WIDTH, 3),
        activation='sigmoid',
        decoder_block_type='transpose',
        encoder_weights='imagenet',
        decoder_use_batchnorm=True
    )

    model.summary()
    model.compile(optimizer=Adadelta(1e-3), loss=loss, metrics=[dice_score, jaccard_score])

    model_json = model.to_json()
    json_file = open('models/linknet_gray' + str(BATCH) + '_batch.json', 'w')
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
