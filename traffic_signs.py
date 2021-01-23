import tensorflow as tf
import tensorflow_datasets as tfds 
import numpy as np
import matplotlib.pyplot as plt 
import cv2

from tensorflow import keras

# Process the data
train_ds, test_ds = tfds.load('visual_domain_decathlon/gtsrb', split = ['train', 'test'], as_supervised = True)

train_ds = tfds.as_numpy(train_ds)
test_ds = tfds.as_numpy(test_ds)

def create_img_label_set(data_set):
    image_set = []
    label_set = []
    for img in data_set:
       image, label = img[0], img[1]
       image_set.append(image)
       label_set.append(label)
    return np.asarray(image_set), np.asarray(label_set)

img_train, label_train = create_img_label_set(train_ds)
img_test, label_test = create_img_label_set(test_ds)

print('Train images size:', img_train.size)
print('Train labels shape:', label_train.shape)
print('Test images shape:', img_test.shape)
print('Test labels shape:', label_test.shape)

# Build the model
#model = keras.Sequential()
#model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(64,64,3)))
#model.add(keras.layers.Activation('relu'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(keras.layers.Conv2D(32, (3, 3)))
#model.add(keras.layers.Activation('relu'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(keras.layers.Conv2D(64, (3, 3)))
#model.add(keras.layers.Activation('relu'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(64))
#model.add(keras.layers.Activation('relu'))
#model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Dense(2, activation='softmax'))

# Compile the model
#model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
#model.fit(img_train, label_train, validation_data = (img_test, label_test), epochs = 10, batch_size = 64)

# Evaluate the model
#accuracy = model.evaluate(img_test, label_test, verbose = 0)
#print('Accuracy: %.2f%%' % (accuracy[1]*100))

# Make predictions

