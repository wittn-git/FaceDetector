import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from modules.data.data_preprocessing import preprocess_folder
import random

def batch_generator(path_pos: str, path_neg: str, batch_size: int):

    pos_files, neg_files = os.listdir(path_pos), os.listdir(path_neg)

    occs = {k : 0 for k in pos_files + neg_files}

    pos_ratio = len(pos_files) / (len(pos_files) + len(neg_files))
    num_pos_in_batch = int(batch_size * pos_ratio)
    num_neg_in_batch = batch_size - num_pos_in_batch
    pos_idx, neg_idx = 0, 0

    while True:

        batch = []
        labels = np.concatenate([np.ones(num_pos_in_batch), np.zeros(num_neg_in_batch)])
        
        for _ in range(num_pos_in_batch):
            if pos_idx >= len(pos_files):
                pos_idx = 0
                random.shuffle(pos_files)
            occs[pos_files[pos_idx]] += 1
            batch.append(np.load(os.path.join(path_pos, pos_files[pos_idx])))
            pos_idx += 1
        
        for _ in range(num_neg_in_batch):
            if neg_idx >= len(neg_files):
                neg_idx = 0
                random.shuffle(neg_files)
            occs[neg_files[neg_idx]] += 1
            batch.append(np.load(os.path.join(path_neg, neg_files[neg_idx])))
            neg_idx += 1

        yield np.array(batch), labels
    
def build_model(
    train_pos_dir: str,
    train_neg_dir: str,
    model_path: str,
    batch_size: int = 32,
    epochs: int = 10
):
    
    assert model_path.endswith(".keras") 

    train_generator = batch_generator(train_pos_dir, train_neg_dir, batch_size)

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
    print("Compiled model")
    model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=  10 * batch_size,
        epochs=epochs
    )
    
    model.save(model_path)

def test_model(model_path : str, threshold : float, test_pos_dir : str, test_neg_dir : str, preprocessed : bool) -> "tuple[int, float, int, float]":

    model = load_model(model_path)

    def load_data(folder_path: str):
        data = []
        for filename in os.listdir(folder_path):
            img = np.load(os.path.join(folder_path, filename))
            data.append(img)
        return np.array(data)
    
    if preprocessed:
        X_pos, X_neg = load_data(test_pos_dir), load_data(test_neg_dir)
    else:
        X_pos, X_neg = preprocess_folder(test_pos_dir, (80, 80)), preprocess_folder(test_neg_dir, (80, 80))
   
    y_pos, y_neg = np.ones(len(X_pos)), np.zeros(len(X_neg))
    y_pos_pred, y_neg_pred = model.predict(X_pos), model.predict(X_neg)
    y_pos_pred, y_neg_pred = (y_pos_pred > threshold).astype(int), (y_neg_pred > threshold).astype(int)

    pos_accuracy, neg_accuracy = accuracy_score(y_pos, y_pos_pred), accuracy_score(y_neg, y_neg_pred)

    return len(y_pos), pos_accuracy, len(y_neg), neg_accuracy