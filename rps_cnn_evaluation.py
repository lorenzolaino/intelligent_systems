import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names

proj_dir_path = 'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto ' \
                'Laino\\intelligent_systems'

# Load json and create model
file = open(f'{proj_dir_path}\\trained_models\\rps_10_epochs_cnn_model.json', 'r')
json_model = file.read()
file.close()
loaded_model = keras.models.model_from_json(json_model)

# Load weights
loaded_model.load_weights(f'{proj_dir_path}\\trained_models\\rps_10_epochs_cnn_model_weights.hdf5')

predictions = loaded_model.predict(img_test)


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
