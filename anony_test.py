import cv2
import numpy as np
import face_recognition
import pickle
from annoy import AnnoyIndex


img = face_recognition.load_image_file("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(img)[0]


f = 128
u = AnnoyIndex(f, "euclidean")
u.load("test.ann")

with open("refer.pkl", "rb") as f:
    refes = pickle.load(f)

nearest = u.get_nns_by_vector(test_encode, 1, include_distances=True)

print(nearest)
print(f"This is {refes[nearest[0][0]]}")
