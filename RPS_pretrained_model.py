import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB2

# Process the data
train_ds, test_ds = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True)

train_ds = tfds.as_numpy(train_ds)
test_ds = tfds.as_numpy(test_ds)


def create_img_label_set(data_set, dimension):
    image_set = []
    label_set = []
    for img in data_set:
        image, label = img[0], img[1]
        res_img = cv2.resize(image, dsize=(dimension, dimension), interpolation=cv2.INTER_CUBIC)
        image_set.append(res_img)
        # image_set.append(image)
        label_set.append(label)
    return np.asarray(image_set), np.asarray(label_set)


img_train, label_train = create_img_label_set(train_ds, 50)
img_test, label_test = create_img_label_set(test_ds, 50)
print('Total images', img_train.shape[0] + img_test.shape[0])

print('**', label_train)
unique_label_values = np.unique(label_train)
print('Labels number:', len(unique_label_values))
print('Train images shape:', img_train.shape)
print('Train labels shape:', label_train.shape)
print('Test images shape:', img_test.shape)
print('Test labels shape:', label_test.shape)


def build_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(50, 50, 3))
    # inputs = tf.keras.layers.Input(shape=(300, 300, 3))
    model = EfficientNetB2(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


model = build_model(num_classes=3)
epochs = 2  # @param {type: "slider", min:8, max:80}
hist = model.fit(img_train, label_train, epochs=epochs, validation_data=(img_test, label_test), verbose=2)
plot_hist(hist)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))

# Make predictions
class_names = ['Rock', 'Paper', 'Scissors']

predictions = model.predict(img_test)


def plot_image(i):
    img = img_test[i]
    predicted = predictions[i]
    label = label_test[i]

    predicted_label = np.argmax(predicted)
    color = 'red' if predicted_label != label else 'green'

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label], 100 * np.max(predicted), class_names[label]),
               color=color)


for i in range(0, 4):
    rand = np.random.randint(0, len(img_test))
    print(rand)
    plt.figure(figsize=(6, 3))
    plot_image(rand)
    plt.show()
