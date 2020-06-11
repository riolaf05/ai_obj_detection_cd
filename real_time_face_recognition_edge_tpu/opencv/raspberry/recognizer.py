import cv2
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np 
import pickle
import RPi.GPIO as GPIO
from time import sleep

BASE_DIR=r'C:\Users\lafacero\Desktop\opencv\x86'

relay_pin = [26]
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, 0)

with open(os.path.join(BASE_DIR,'labels'), 'rb') as f:
	dicti = pickle.load(f)
	f.close()

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR,"trainer.yml"))

font = cv2.FONT_HERSHEY_SIMPLEX

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roiGray)

        #for name, value in dict.items():
        #    if value == id_:
        #        print(name)
        
        if conf <= 70:
            print("ok")
            #TODO: add on screen labels here
        else:
            print("no")
        
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == 27:
        break

cv2.destroyAllWindows()