from tensorflow import keras

from rps_process_data import RPSProcessData

RPS_process_data = RPSProcessData()
img_train, label_train, img_test, label_test = RPS_process_data.train_test_split()
label_names = RPS_process_data.label_names
RPS_process_data.print_dataset_shapes()

# Build the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(50, 50, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(label_names), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
model.fit(img_train, label_train, validation_data=(img_test, label_test), epochs=10, batch_size=105)

# Evaluate the model
accuracy = model.evaluate(img_test, label_test, verbose=0)
print(model.metrics_names)
print('Accuracy: %.2f%%' % (accuracy[1] * 100))

# Save the model
json_path_file = 'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto Laino\\intelligent_systems\\rps_50_cnn_model.json'
hdf5_path_file = 'D:\\Utenti\\Marco\\Desktop\\Insubria\\Laurea Magistrale\\Intelligent Systems\\Exam Project Gatto Laino\\intelligent_systems\\rps_50_cnn_model_weights.hdf5'
json_file = model.to_json()
with open(json_path_file, 'w') as file:
    file.write(json_file)

# Serialize weights to HDF5
model.save_weights(hdf5_path_file)
