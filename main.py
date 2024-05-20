import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file('emma.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# face = face_recognition.face_locations(img)[0]
train_encode = face_recognition.face_encodings(img)[0]


test = face_recognition.load_image_file('gal.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

test_encode = face_recognition.face_encodings(test)[0]

print(face_recognition.compare_faces([train_encode],test_encode))

print(train_encode)
