import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names
RPS_process_data.print_dataset_shapes()


def build_efn_b0_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(50, 50, 3))
    efn_b0_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    efn_b0_model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(efn_b0_model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    efn_b0_model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    efn_b0_model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return efn_b0_model


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


model = build_efn_b0_model(num_classes=len(label_names))
epochs = 2
hist = model.fit(img_train, label_train, epochs=epochs, validation_data=(img_test, label_test), verbose=2)
plot_hist(hist)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))

# Make predictions
predictions = model.predict(img_test)


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
