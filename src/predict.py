import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('../models/image_classifier.h5')


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image / 255.0  # normalizar
    return np.expand_dims(image, axis=0)


def predict(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    return class_idx

if __name__ == "__main__":
    image_path = '../data/custom_images/sample_image.jpg'
    result = predict(image_path)
    print(f"Classe prevista: {result}")
