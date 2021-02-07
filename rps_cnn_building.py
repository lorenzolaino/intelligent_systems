import matplotlib.pyplot as plt

from tensorflow import keras

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names
RPS_process_data.print_dataset_shapes()

# Build the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(50, (3, 3), input_shape=(50, 50, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(25, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(50, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(label_names), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# Train the model
hist = model.fit(img_train, label_train, validation_data=(img_test, label_test), epochs=10, batch_size=105)
plot_hist(hist)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test, verbose=0)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))

proj_dir_path = 'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto ' \
                'Laino\\intelligent_systems'

# Save the model
json_file = model.to_json()
with open(f'{proj_dir_path}\\trained_models\\rps_10_epochs_cnn_model.json', 'w') as file:
    file.write(json_file)

# Serialize weights to HDF5
model.save_weights(f'{proj_dir_path}\\trained_models\\rps_10_epochs_cnn_model_weights.hdf5')
