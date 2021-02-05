import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.efficientnet import EfficientNetB4

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names
RPS_process_data.print_dataset_shapes()
RPS_process_data.plot_rnd_imgs()


def build_efn_b4_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(50, 50, 3))
    efn_b4_model = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    efn_b4_model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(efn_b4_model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    efn_b4_model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    efn_b4_model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return efn_b4_model


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


model = build_efn_b4_model(num_classes=len(label_names))
epochs = 5
hist = model.fit(img_train, label_train, epochs=epochs, validation_data=(img_test, label_test),
                 verbose=2, batch_size=105)
plot_hist(hist)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))

# Make predictions
predictions = model.predict(img_test)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    # Normalize the confusion matrix.
    conf_matrix = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        color = "white" if conf_matrix[i, j] > threshold else "black"
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Use the model to predict the values from the test_images.
test_pred = np.argmax(predictions, axis=1)

# Calculate the confusion matrix using sklearn.metrics
cm = confusion_matrix(label_test, test_pred)

plot_confusion_matrix(cm)


def plot_image(i):
    img = img_test[i]
    predicted = predictions[i]
    label = label_test[i]

    predicted_label = np.argmax(predicted)
    color = 'red' if predicted_label != label else 'green'

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel('{} {:2.0f}% ({})'.format(label_names[predicted_label], 100 * np.max(predicted), label_names[label]),
               color=color)


for i in range(0, 4):
    rand = np.random.randint(0, len(img_test))
    print(rand)
    plt.figure(figsize=(7, 4))
    plot_image(rand)
    plt.show()
