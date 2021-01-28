import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
#import efficientnet.keras as efn
import cv2

from tensorflow import keras
from efficientnet.keras import EfficientNetB2

# Process the data
train_ds, test_ds = tfds.load('rock_paper_scissors', split = ['train', 'test'], as_supervised = True)

train_ds = tfds.as_numpy(train_ds)
test_ds = tfds.as_numpy(test_ds)

def create_img_label_set(data_set, dimension):
    image_set = []
    label_set = []
    for img in data_set:
        image, label = img[0], img[1]
        res_img = cv2.resize(image, dsize = (dimension ,dimension), interpolation = cv2.INTER_CUBIC)
        image_set.append(res_img)
        label_set.append(label)
    return np.asarray(image_set), np.asarray(label_set)

img_train, label_train = create_img_label_set(train_ds, 30)
img_test, label_test = create_img_label_set(test_ds, 30)
print('Total images', img_train.shape[0] + img_test.shape[0])

print('**', label_train)
unique_label_values = np.unique(label_train)
print('Labels number:', len(unique_label_values))
print('Train images shape:', img_train.shape)
print('Train labels shape:', label_train.shape)
print('Test images shape:', img_test.shape)
print('Test labels shape:', label_test.shape)

# Import efficientnet
model = EfficientNetB2(weights = 'imagenet')
