import os
import cv2
import numpy as np

def preprocess_folder(source_folder_path: str, img_size=(80, 80), target_folder_path : str = None):
    if target_folder_path is not None and not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    imgs = []
    for filename in os.listdir(source_folder_path):
        img = cv2.imread(os.path.join(source_folder_path, filename))
        filename = filename[:-4] + ".npy"
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.resize(img, img_size) 
            img = img.astype(np.float32) / 255.0  
            if target_folder_path is not None:
                np.save(os.path.join(target_folder_path, filename), img)
            else:
                imgs.append(img)
    if target_folder_path is None:
        return np.array(imgs)