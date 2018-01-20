
"""
Author: Jeffrey Valentic
Date: 1/20/2018
File: get_cam_feed.py
Version 1.0
Notes:
    -This is a script that will save a frame from the webcame every 20 loops
    -Installs:
        ~ OpenCV
        ~ numpy
"""
import numpy as np
import cv2

#0 is default camera for VideoCapture arg
cap = cv2.VideoCapture(0)

#How often we want to get a picture
fps = 20
current = 0

#Will run until "q" key is pressed.
while(True):
    #Only captures image if it has gone through 20 loops.
    if (current % fps == 0):
        #Resets current (current is the current loop number) to 0.
        current = 0
        # Capture frame-by-frame (frame is the actual image)
        ret, frame = cap.read()
    
    #waitKey(1) will wait 1 milisecond for the break key (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        break
    current += 1


