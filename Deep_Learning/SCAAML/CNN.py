import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

plaintext = np.load('C:/Users/부채널분석1/Desktop/SCAAML/2022.03.25-05.40.47_textin.npy')

hw_iv = [[0 for _ in range(1)] for _ in range(10000)]

for i in range(10000):
    iv = plaintext[i][0]
        
    for k in range(8):
        hw_iv[i][0] += ((iv>>k) & 1)

np.save('C:/Users/부채널분석1/Desktop/SCAAML/HW_IV.npy', hw_iv)

trace = np.load('C:/Users/부채널분석1/Desktop/SCAAML/2022.03.25-05.40.47_traces.npy.bak')
trace = trace[:,:7200]
hamming_weight = np.load('C:/Users/부채널분석1/Desktop/SCAAML/HW_IV.npy')
x_train, x_test = train_test_split(trace, test_size = 0.2, shuffle = False)
y_train, y_test = train_test_split(hamming_weight, test_size = 0.2, shuffle = False)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

model = keras.Sequential(
    [
        keras.Input(shape=(7200, 1)),
        layers.Conv1D(64, kernel_size=(32), activation="relu", padding = 'same'),
        layers.MaxPooling1D(pool_size=(10)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv1D(32, kernel_size=(32), activation="relu", padding = 'same'),
        layers.MaxPooling1D(pool_size=(10)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv1D(16, kernel_size=(32), activation="relu", padding = 'same'),
        layers.MaxPooling1D(pool_size=(10)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(9, activation="softmax"),
    ]
)

model.summary()

batch_size = 32
epochs = 50

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.2, verbose = 1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Loss:", score[0])
print("Accuracy:", score[1])
