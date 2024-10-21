import os
import cv2
import numpy as np

def load_images_from_folder(folder, target_size=(32, 32)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img / 255.0 
            images.append(img)
            labels.append(0)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    images, labels = load_images_from_folder('../data/custom_images')
    print(f"Carregadas {len(images)} imagens.")
