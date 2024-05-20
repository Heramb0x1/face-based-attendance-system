import cv2
import numpy as np
import face_recognition
import pickle


img = face_recognition.load_image_file("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test_encode = face_recognition.face_encodings(img)[0]

with open("encodings.pkl", "rb") as f:
    faces = pickle.load(f)


for face in faces:
    if face_recognition.compare_faces([faces[face]], test_encode)[0]:
        print(f"This is {face}")
