import os
from PIL import Image
import numpy as np

def generate_copies(img : Image.Image, num_copies : int, attribute_prob = 0.5) -> "list[Image.Image]":

    copies = []

    for _ in range(num_copies):
        copy = np.array(img.copy())
        if np.random.rand() > attribute_prob:
            copy = np.rot90(copy, np.random.randint(4))
        if np.random.rand() > attribute_prob:
            copy = np.flip(copy, axis=0)
        if np.random.rand() > attribute_prob:
            copy = np.flip(copy, axis=1)
        if np.random.rand() > attribute_prob:
            copy = copy.astype(np.float64)
            copy += np.random.normal(0, 0.1, copy.shape)
            copy = np.clip(copy, 0, 255)
            copy = copy.astype(np.uint8)
        if np.random.rand() > attribute_prob:
            copy = np.clip(copy * np.random.uniform(0.5, 1.5), 0, 255).astype(np.uint8)
        if np.random.rand() > attribute_prob:
            x = np.random.randint(0, copy.shape[0]//2)
            y = np.random.randint(0, copy.shape[1]//2)
            copy = copy[x:x+copy.shape[0]//2, y:y+copy.shape[1]//2]

        copy = Image.fromarray(copy)
        copy = copy.resize((80, 80))
        copies.append(copy)

    return copies

def generate_data(folder_path : str, num_copies : int = 10, num_files : int = None):
    
    files = os.listdir(folder_path)
    if num_files is None:
        num_files = len(files)
    files = files[:num_files]

    for file in files:
        img = Image.open(os.path.join(folder_path, file))
        copies = generate_copies(img, num_copies)
        for i, copy in enumerate(copies):
            copy.save(os.path.join(folder_path, f"{file[:-4]}_{i}.jpg"))