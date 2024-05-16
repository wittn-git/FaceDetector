import os
import cv2
import numpy as np

def preprocess_folder(source_folder_path: str, target_folder_path : str, img_size=(80, 80)):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    print(len(os.listdir(source_folder_path)))
    for filename in os.listdir(source_folder_path):
        img = cv2.imread(os.path.join(source_folder_path, filename))
        filename = filename[:-4] + ".npy"
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.resize(img, img_size) 
            img = img.astype(np.float32) / 255.0  
            np.save(os.path.join(target_folder_path, filename), img)