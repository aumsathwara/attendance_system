import cv2
import os
import pandas as pd
import time
import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
recognizer.read("TrainingImageLabel" + os.sep + "Trainner.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")
font = cv2.FONT_HERSHEY_SIMPLEX
col_names = ['Id', 'Name', 'Date', 'Time']
attendance = pd.DataFrame(columns=col_names)
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
ret, im = cam.read()
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in faces:
    cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
print( df.loc["Id"].values)