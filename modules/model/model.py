import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

def get_data(path : str, positive : bool):
    def load_data(folder):
        data = []
        for filename in os.listdir(folder):
            if filename.endswith('.npy'):
                filepath = os.path.join(folder, filename)
                image = np.load(filepath)
                data.append(image)
        return np.array(data)
    X = load_data(path)
    if positive:
        return X, np.ones(X.shape[0])
    return X, np.zeros(X.shape[0])


def build_model(
    train_pos_dir: str,
    train_neg_dir: str,
    model_path: str,
    validation_split: float = 0.2,
    batch_size: int = 32,
    epochs: int = 10
):

    X_pos, y_pos = get_data(train_pos_dir, True)
    X_neg, y_neg = get_data(train_neg_dir, True)

    X_train = np.concatenate((X_pos, X_neg), axis=0)
    y_train = np.concatenate((y_pos, y_neg), axis=0)

    shuffle_index = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    model.save(model_path)

def test_model(model_path : str, threshold : float, test_pos_dir : str, test_neg_dir : str) -> "tuple[int, float, int, float]":

    model = load_model(model_path)

    X_pos, y_pos = get_data(test_pos_dir, True)
    y_pos_pred = model.predict(X_pos)
    y_pos_pred = (y_pos_pred > threshold).astype(int)
    X_neg, y_neg = get_data(test_neg_dir, True)
    y_neg_pred = model.predict(X_neg)
    y_neg_pred = (y_neg_pred > threshold).astype(int)

    pos_accuracy = accuracy_score(y_pos, y_pos_pred)
    neg_accuracy = accuracy_score(y_neg, y_neg_pred)

    return len(y_pos), pos_accuracy, len(y_neg), neg_accuracy