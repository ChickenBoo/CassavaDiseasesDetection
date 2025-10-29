import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (244, 244))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image
