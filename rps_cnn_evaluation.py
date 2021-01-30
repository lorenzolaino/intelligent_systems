import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names

# Load json and create model
file = open(
    'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto Laino\\intelligent_systems\\rps_50_cnn_model.json',
    'r')
json_model = file.read()
file.close()

loaded_model = keras.models.model_from_json(json_model)
# Load weights
loaded_model.load_weights(
    'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto Laino\\intelligent_systems\\rps_50_cnn_model_weights.hdf5')

predictions = loaded_model.predict(img_test)


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
