from segmentation_models.linknet import Linknet
from segmentation_models.utils import set_trainable
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
import h5py
from segmentation_models.metrics import dice_score, jaccard_score
from segmentation_models.losses import cce_dice_loss, jaccard_loss

SEED = 42
smooth = 1e-10
HEIGHT, WIDTH, DEPTH = 224, 224, 3
IMAGES = 'E:/datasets/parking/images'
MASKS = 'E:/datasets/parking/masks'
BATCH = 8
CLASSES = {
    'car': 76,
    'road': 29,
    'line': 255
}


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

    dataset_name = 'birdEyeViewSemantic.hdf5'

    if os.path.isfile(dataset_name):
        data = h5py.File(dataset_name, 'r')
        print('read dataset from hdf5')
        return data['images'][()], data['masks'][()]

    images = os.listdir(IMAGES)
    masks = os.listdir(MASKS)

    x_data = np.empty((len(images), HEIGHT, WIDTH, 3), dtype=np.uint8)
    y_data = np.empty((len(masks), HEIGHT, WIDTH, len(CLASSES)), dtype=np.uint8)

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
            temp[temp != 0] = 255
            np.append(mask_list, temp)

        x_data[i] = image
        y_data[i] = mask_list

    print(f'{len(x_data)} images loaded!')

    data = h5py.File(dataset_name, 'w')

    data.create_dataset('images', data=x_data)
    data.create_dataset('masks', data=y_data)
    data.close()

    return x_data, y_data


if __name__ == '__main__':
    x_data, y_data = prepare_data()

    train_images, val_images, train_masks, val_masks = train_test_split(x_data, y_data, test_size=0.2,
                                                                        random_state=SEED, shuffle=True)
    callbacks_list = [
        ModelCheckpoint('models/linknet' + str(len(CLASSES)) + '_classes.h5',
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
        input_shape=(HEIGHT, WIDTH, DEPTH),
        classes=len(CLASSES),
        activation='sigmoid',
        decoder_block_type='upsampling',
        encoder_weights='imagenet',
        encoder_freeze=True,
        decoder_use_batchnorm=True
    )

    model.summary()
    model.compile(optimizer=Adam(1e-3), loss=jaccard_loss, metrics=[dice_score, jaccard_score])

    model_json = model.to_json()
    json_file = open('models/linknet' + str(len(CLASSES)) + '_classes.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    # 1st stage
    model.fit_generator(
        my_generator(train_images, train_masks, BATCH),
        steps_per_epoch=len(train_masks) / BATCH,
        epochs=5,
        verbose=1,
        validation_data=my_generator(val_images, val_masks, 1),
        validation_steps=len(val_images)
    )

    set_trainable(model)

    # 2nd stage
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