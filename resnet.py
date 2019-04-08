import keras.backend as K
from segmentation_models import Linknet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from skimage.morphology import remove_small_objects
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

SEED = 42
smooth = 1e-10


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        rotation_range=30,
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        rotation_range=30,
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def val_generator(x_train, y_train, batch_size=1):
    data_generator = ImageDataGenerator(
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def prepare_images(train_images_path):
    print('preparing images...')
    # get names of jpg files inside folder and create a list
    train_images = list(filter(lambda x: x.endswith('.jpg'), os.listdir(train_images_path)))[:1000]

    # input data array
    x_data = np.empty((len(train_images), image_h, image_w, 3), dtype='uint8')
    tbar = tqdm(train_images)
    for i, file_name in enumerate(tbar):
        img = cv2.imread(os.path.join(train_images_path, file_name), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(image_w, image_h))
        x_data[i] = img

    return x_data


def prepare_masks(train_masks_path):
    print('preparing masks...')
    # get names of png files inside folder and create a list
    train_masks = list(filter(lambda x: x.endswith('.png'), os.listdir(train_masks_path)))[:1000]

    # output data array
    y_data = np.empty((len(train_masks), image_h, image_w, 1), dtype='uint8')
    tbar = tqdm(train_masks)
    for i, file_name in enumerate(tbar):
        img = cv2.imread(os.path.join(train_masks_path, file_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(image_w, image_h))
        img[img != 255] = 0
        img = img[:, :, np.newaxis]
        y_data[i] = img

    return y_data


def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='Name for model')
    args = ap.parse_args()
    return args


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


if __name__ == '__main__':
    exargs = argparser()
    train_images_path = 'E:/datasets/parking/images'
    train_masks_path = 'E:/datasets/parking/masks'
    image_h = 288
    image_w = 288

    x_data = prepare_images(train_images_path)
    y_data = prepare_masks(train_masks_path)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)

    model = Linknet(backbone_name='mobilenetv2',
                    input_shape=(image_h, image_w, 3),
                    encoder_weights='imagenet',
                    decoder_block_type='transpose',
                    activation='sigmoid')
    model.summary()

    callbacks_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-6)]

    # model.load_weights('../weights/resnet34_RLE_72_loss.h5')

    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, jaccard_coef])

    model_json = model.to_json()
    json_file = open('models/' + exargs.name + '.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    save_name = 'models/' + exargs.name + '.h5'
    save_name_loss = 'models/' + exargs.name + '_loss.h5'
    callbacks_list.append(
        ModelCheckpoint(save_name_loss,
                        verbose=1,
                        monitor='loss',
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True))
    callbacks_list.append(
        ModelCheckpoint(save_name,
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True))
    history = model.fit_generator(my_generator(x_train, y_train, 4),
                                  steps_per_epoch=len(x_train),
                                  validation_data=val_generator(x_val, y_val),
                                  validation_steps=len(x_val),
                                  epochs=10,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=callbacks_list)
    K.clear_session()
