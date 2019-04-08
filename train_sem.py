import os

import cv2
import h5py
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from segmentation_models.linknet import Linknet
from segmentation_models.losses import cce_jaccard_loss
from segmentation_models.metrics import jaccard_score
from tqdm import tqdm

SEED = 42
smooth = 1e-10
HEIGHT, WIDTH, DEPTH = 224, 224, 3
IMAGES = 'E:/datasets/parking/images'
MASKS = 'E:/datasets/parking/masks'
BATCH = 4
CLASSES = {
    'car': 76,
    'road': 29,
    'line': 255
}


def IoU(y_true, y_pred, smooth=100.):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def IoU_loss(y_true, y_pred):
    return 1. - IoU(y_true, y_pred)


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def prepare_data():
    print('starting making data..')

    dataset_name = 'birdEyeViewSemantic_new.hdf5'

    if os.path.isfile(dataset_name):
        data = h5py.File(dataset_name, 'r')
        print('read dataset from hdf5')
        return data['images'][()], data['masks'][()]

    images = os.listdir(IMAGES)
    masks = os.listdir(MASKS)

    x_data = np.empty((len(images), HEIGHT, WIDTH, 3), dtype=np.uint8)
    y_data = np.empty((len(masks), HEIGHT, WIDTH, len(CLASSES)), dtype=np.bool)

    tbar = tqdm(images)
    for i, file_name in enumerate(tbar):
        image = cv2.imread(os.path.join(IMAGES, file_name), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(os.path.join(MASKS, file_name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)

        mask_list = np.empty(shape=(HEIGHT, WIDTH, len(CLASSES)))
        for j, layer in enumerate(CLASSES):
            temp = mask.copy()
            temp[temp != CLASSES[layer]] = 0
            temp[temp != 0] = 1
            mask_list[:, :, j] = temp

        x_data[i] = image
        y_data[i] = mask_list

    print(f'{len(x_data)} images loaded!')

    data = h5py.File(dataset_name, 'w')

    data.create_dataset('images', data=x_data)
    data.create_dataset('masks', data=y_data)
    data.close()

    return x_data, y_data


def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)


if __name__ == '__main__':
    x_data, y_data = prepare_data()

    train_images, val_images, train_masks, val_masks = x_data[:5000], x_data[5000:], y_data[:5000], y_data[5000:]
    callbacks_list = [
        ModelCheckpoint('models/linknet_sem' + str(len(CLASSES)) + '_classes.h5',
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
        backbone_name='resnet34',
        input_shape=(HEIGHT, WIDTH, DEPTH),
        classes=len(CLASSES),
        activation='sigmoid',
        decoder_block_type='transpose',
        encoder_weights='imagenet',
        decoder_use_batchnorm=True
    )

    model.summary()
    model.compile(optimizer=Adam(1e-4), loss=cce_jaccard_loss, metrics=[jaccard_score])

    model_json = model.to_json()
    json_file = open('models/linknet_sem' + str(len(CLASSES)) + '_classes.json', 'w')
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

    # result = val_masks[0, :, :, 2]
    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].imshow(val_masks[0, :, :, 1])
    # axes[0, 1].imshow(val_masks[0, :, :, 2])
    # plt.show()
    # exit()
