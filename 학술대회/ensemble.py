import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
trace_file = 'C:/Users/부채널분석1/Desktop/ches_ctf_2018_traces.npy'  
label_file = 'C:/Users/부채널분석1/Desktop/ches_ctf_2018_labels.npy'   

trace = np.load(trace_file)
labels = np.load(label_file)

x_train, x_test, y_train, y_test = train_test_split(trace, labels, test_size=0.2, shuffle=False)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# MLP 모델 하이퍼파라미터 범위
learning_rates = np.arange(0.0001, 0.0011, 0.0001)
batch_sizes = np.arange(100, 1001, 100)
dense_layers_counts = np.arange(2, 9, 1)
neurons_counts = np.arange(100, 1001, 100)

# CNN 모델 하이퍼파라미터 범위
conv_layers_counts = np.arange(1, 3, 1)
filters_counts = np.arange(8, 33, 4)
kernel_sizes = np.arange(10, 21, 2)
strides = np.arange(5, 11, 5)
dense_layers_counts_cnn = np.arange(2, 4, 1)
neurons_counts_cnn = np.arange(100, 1001, 100)

# MLP 모델 생성
mlp_models = []
for _ in range(50):
    lr = random.choice(learning_rates)
    bs = random.choice(batch_sizes)
    dl = random.choice(dense_layers_counts)
    nc = random.choice(neurons_counts)

    model = Sequential()
    model.add(Dense(nc, activation='relu', input_dim=x_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    for _ in range(dl - 1):
        model.add(Dense(nc, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=bs, epochs=50, verbose=0)
    mlp_models.append(model)

# CNN 모델 생성
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

cnn_models = []
for _ in range(50):
    lr = random.choice(learning_rates)
    bs = random.choice(batch_sizes)
    cl = random.choice(conv_layers_counts)
    f = random.choice(filters_counts)
    ks = random.choice(kernel_sizes)
    s = random.choice(strides)
    dl = random.choice(dense_layers_counts_cnn)
    nc = random.choice(neurons_counts_cnn)

    model = Sequential()
    model.add(Conv1D(filters=f, kernel_size=ks, strides=s, activation='relu', padding='same', input_shape=(x_train_cnn.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    for _ in range(cl - 1):
        model.add(Conv1D(filters=f, kernel_size=ks, strides=s, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Flatten())
    
    for _ in range(dl):
        model.add(Dense(nc, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train_cnn, y_train, batch_size=bs, epochs=50, verbose=0)
    cnn_models.append(model)

# 모델 성능 평가
def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred_classes)
    return accuracy

def predict_with_ensemble(models, x):
    predictions = np.mean([model.predict(x) for model in models], axis=0)
    return np.argmax(predictions, axis=1)

def calculate_guess_entropy(models, x):
    predictions = np.mean([model.predict(x) for model in models], axis=0)
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
    return np.mean(entropy)

# MLP 앙상블 평가
mlp_accuracies = [evaluate_model(model, x_test, y_test) for model in mlp_models]
best_mlp_model = mlp_models[np.argmax(mlp_accuracies)]

top_10_mlp_models = sorted(mlp_models, key=lambda model: evaluate_model(model, x_test, y_test), reverse=True)[:10]
ensemble_50_mlp_predictions = predict_with_ensemble(mlp_models, x_test)
mlp_best_accuracy = evaluate_model(best_mlp_model, x_test, y_test)
mlp_top_10_accuracy = accuracy_score(y_test, predict_with_ensemble(top_10_mlp_models, x_test))
mlp_ensemble_50_accuracy = accuracy_score(y_test, ensemble_50_mlp_predictions)

# CNN 앙상블 평가
cnn_accuracies = [evaluate_model(model, x_test_cnn, y_test) for model in cnn_models]
best_cnn_model = cnn_models[np.argmax(cnn_accuracies)]

top_10_cnn_models = sorted(cnn_models, key=lambda model: evaluate_model(model, x_test_cnn, y_test), reverse=True)[:10]
ensemble_50_cnn_predictions = predict_with_ensemble(cnn_models, x_test_cnn)
cnn_best_accuracy = evaluate_model(best_cnn_model, x_test_cnn, y_test)
cnn_top_10_accuracy = accuracy_score(y_test, predict_with_ensemble(top_10_cnn_models, x_test_cnn))
cnn_ensemble_50_accuracy = accuracy_score(y_test, ensemble_50_cnn_predictions)

# 성능 그래프
plt.figure(figsize=(14, 7))

# MLP 성능 비교
plt.subplot(1, 2, 1)
plt.bar(['Best Model', 'Top 10 Ensemble', '50 Model Ensemble'],
        [mlp_best_accuracy, mlp_top_10_accuracy, mlp_ensemble_50_accuracy])
plt.title('MLP Performance Comparison')
plt.ylabel('Accuracy')

# CNN 성능 비교
plt.subplot(1, 2, 2)
plt.bar(['Best Model', 'Top 10 Ensemble', '50 Model Ensemble'],
        [cnn_best_accuracy, cnn_top_10_accuracy, cnn_ensemble_50_accuracy])
plt.title('CNN Performance Comparison')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# 추측 엔트로피 비교
plt.figure(figsize=(14, 7))

# MLP 추측 엔트로피
plt.subplot(1, 2, 1)
plt.bar(['Best Model', 'Top 10 Ensemble', '50 Model Ensemble'],
        [calculate_guess_entropy([best_mlp_model], x_test),
         calculate_guess_entropy(top_10_mlp_models, x_test),
         calculate_guess_entropy(mlp_models, x_test)])
plt.title('MLP Guess Entropy Comparison')
plt.ylabel('Guess Entropy')

# CNN 추측 엔트로피
plt.subplot(1, 2, 2)
plt.bar(['Best Model', 'Top 10 Ensemble', '50 Model Ensemble'],
        [calculate_guess_entropy([best_cnn_model], x_test_cnn),
         calculate_guess_entropy(top_10_cnn_models, x_test_cnn),
         calculate_guess_entropy(cnn_models, x_test_cnn)])
plt.title('CNN Guess Entropy Comparison')
plt.ylabel('Guess Entropy')

plt.tight_layout()
plt.show()
