import cv2
import face_recognition
import os
import pickle

print(os.listdir("training"))
transet = os.listdir("training")
encodings = {}


for img in transet:
    name = img[:-4]
    imgf = face_recognition.load_image_file(f"training/{img}")
    imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)

    train_encode = face_recognition.face_encodings(imgf)[0]
    encodings[name] = train_encode

with open("encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)
