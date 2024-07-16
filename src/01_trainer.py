import os

import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = '../dataset'


def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for image_path in image_paths:
        faceImg = Image.open(image_path).convert('L')  # L for luminance
        faceNp = np.array(faceImg, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow('Training', faceNp)
        cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = get_images(path)
recognizer.train(faces, np.array(ids))
recognizer.save('../recognizer/trainingData.yml')
cv2.destroyAllWindows()
