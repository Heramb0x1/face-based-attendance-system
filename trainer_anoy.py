import cv2
import face_recognition
import os
import pickle
from annoy import AnnoyIndex

print(os.listdir("training"))
transet = os.listdir("training")
encodings = {}
f = 128
t = AnnoyIndex(f, "angular")


for img in transet:
    name = img[:-4]
    imgf = face_recognition.load_image_file(f"training/{img}")
    imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)

    train_encode = face_recognition.face_encodings(imgf)[0]
    encodings[name] = train_encode

referal = {}

for inx, nam in enumerate(encodings):
    t.add_item(inx, encodings[nam])
    referal[inx] = nam
    # print(f"{inx} :: {encodings[nam]}")

t.build(10)
t.save("test.ann")


with open("refer.pkl", "wb") as f:
    pickle.dump(referal, f)
