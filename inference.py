import numpy as np
import os
import cv2
from keras.optimizers import Adam
from segmentation_models.losses import bce_dice_loss
from segmentation_models.metrics import dice_score, jaccard_score
from keras.models import model_from_json

HEIGHT, WIDTH, DEPTH = 224, 224, 3

if __name__ == '__main__':
    json = 'models/linknet16.json'
    weight = 'models/linknet16.h5'

    json = open(json, 'r')

    model = model_from_json(json.read())
    model.load_weights(weight)

    model.compile(optimizer=Adam(1e-3), loss=bce_dice_loss, metrics=[dice_score, jaccard_score])

    image = cv2.imread('image_0.jpg', cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = image.reshape(1, HEIGHT, WIDTH, DEPTH)

    predict = model.predict(image)

    result = predict[0, :, :, 0]

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
