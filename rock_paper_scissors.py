import tensorflow as tf
import tensorflow_datasets as tfds 
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow import keras

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

#plt.imshow(img_train[0])
#plt.show()
#print(img_train[0].shape)
#print(img_train[1].shape)
#print(img_train[23].shape)
#print(img_train[5].shape)
#print(img_train[2].shape)
#print(img_train[29].shape)

print('**', label_train)
unique_label_values = np.unique(label_train)
print('Labels number:', len(unique_label_values))
print('Train images shape:', img_train.shape)
print('Train labels shape:', label_train.shape)
print('Test images shape:', img_test.shape)
print('Test labels shape:', label_test.shape)

# Build the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), input_shape = (30, 30, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(64)) #292*292 is the dimension after the 2 MaxPooling
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(unique_label_values), activation='softmax'))
#model.add(keras.layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
model.fit(img_train, label_train, validation_data = (img_test, label_test), epochs = 10, batch_size = 105)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test, verbose = 0)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1]*100))

# Save the model
json_file = model.to_json()
with open('/Users/lorenzolaino/llDev/intelligent_systems/prova.json', 'w') as file:
    file.write(json_file)

# Serialize weights to HDF5
model.save_weights('/Users/lorenzolaino/llDev/intelligent_systems/prova_weights.hdf5')