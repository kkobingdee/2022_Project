import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

plaintext = np.load('C:/Users/부채널분석1/Desktop/SCAAML/aes_trace5000/2022.03.12-04.08.12_textin.npy.bak')

hw_iv = [[0 for _ in range(1)] for _ in range(5000)]

for i in range(5000):
    iv = plaintext[i][0]
        
    for k in range(8):
        hw_iv[i][0] += ((iv>>k) & 1)

np.save('C:/Users/부채널분석1/Desktop/SCAAML/aes_trace5000/HW_IV.npy', hw_iv)

trace = np.load('C:/Users/부채널분석1/Desktop/SCAAML/aes_trace5000/2022.03.12-04.08.12_traces.npy.bak')
hamming_weight = np.load('C:/Users/부채널분석1/Desktop/SCAAML/aes_trace5000/HW_IV.npy')
x_train, x_test = train_test_split(trace, test_size = 0.2, shuffle = False)
y_train, y_test = train_test_split(hamming_weight, test_size = 0.2, shuffle = False)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

model = keras.Sequential(
    [
        layers.Dense(128, activation = 'relu', input_dim = x_train.shape[1]),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation = 'relu'),layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation = 'relu'),layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation = 'relu'),layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(9, activation="softmax"),
    ]
)

model.summary()

batch_size = 64
epochs = 100

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", 
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

#score = model.evaluate(trace, hamming_weight, verbose=0)
#print("Loss:", score[0])
#print("Accuracy:", score[1])
