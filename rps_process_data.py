import cv2
import numpy as np
import tensorflow_datasets as tfds


def create_img_label_set(data_set, dimension):
    image_set = []
    label_set = []
    for img in data_set:
        image, label = img[0], img[1]
        res_img = cv2.resize(image, dsize=(dimension, dimension), interpolation=cv2.INTER_CUBIC)
        image_set.append(res_img)
        label_set.append(label)
    return np.asarray(image_set), np.asarray(label_set)


class RPSProcessData:

    def __init__(self):
        train_ds, test_ds = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True)
        train_ds = tfds.as_numpy(train_ds)
        test_ds = tfds.as_numpy(test_ds)
        img_train, label_train = create_img_label_set(train_ds, 50)
        img_test, label_test = create_img_label_set(test_ds, 50)
        self.img_train = img_train
        self.label_train = label_train
        self.img_test = img_test
        self.label_test = label_test
        self.label_names = ['Rock', 'Paper', 'Scissors']

    def train_test_split(self):
        return self.img_train, self.label_train, self.img_test, self.label_test

    def label_names(self):
        return self.label_names

    def print_dataset_shapes(self):
        print('Labels number:', len(self.label_names))
        print('Total images', self.img_train.shape[0] + self.img_test.shape[0])
        print('Train images shape:', self.img_train.shape)
        print('Train labels shape:', self.label_train.shape)
        print('Test images shape:', self.img_test.shape)
        print('Test labels shape:', self.label_test.shape)
