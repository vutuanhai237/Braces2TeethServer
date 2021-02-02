import cv2, os
import numpy as np

def detect(frame):
    
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml')
    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')
    ds_factor = 1
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imwrite('results/result.png', frame)
    return  '{} {} {} {}'.format(x, y, w, h)