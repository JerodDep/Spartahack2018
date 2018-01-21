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
import predict
import tensorflow as tf
import time
import os

# How often we want to get a picture
fps = 20
current = 0

# Load tensorflow session----------------------------------------------------
## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('trained-model2.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

violence_marks = 0
decision_count = 1
nv_count = 0
v_count = 0

VID_DIR = './presentation_videos'
VIOLENT = '/violent'
NON_VIOLENT = '/non-violent'


#Load first vid
cap = cv2.VideoCapture(VID_DIR + VIOLENT + '/v.0.avi')

#Load alternating violent and non-violent videos
def load_vid(vid_num, nv_count, v_count):

    if vid_num % 2 == 0:
        cap = cv2.VideoCapture(VID_DIR + VIOLENT + '/v.' + str(v_count) + '.avi')
    else:
        cap = cv2.VideoCapture(VID_DIR + NON_VIOLENT + '/nv.' + str(nv_count) + '.avi')

    return cap


clear = lambda: os.system('cls')

start = time.time() # set first start time

violent = False

clear()

# Get Camera feed
while (True):



    #see if it's been x amount of time, then load a new video, alternating between violent and non
    if time.time() - start >= 20:
        cap.release()
        if decision_count % 2 == 0:
            cap = load_vid(decision_count, nv_count, v_count)
            nv_count += 1
        else:
            cap = load_vid(decision_count, nv_count, v_count)
            v_count += 1
        decision_count += 1

        if decision_count > 5:
            decision_count = 0
        if nv_count > 1:
            nv_count = 0
        if v_count > 2:
            v_count = 0

        start = 0
        start = time.time()

    # Only captures image if it has gone through 20 loops.
    if (current % fps == 0):
        # Resets current (current is the current loop number) to 0.
        current = 0
        # Capture frame-by-frame (frame is the actual image)
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        result = predict.prediction(frame, sess, y_true, y_pred, x)

        if (result.item(0) > .6):
            violence_marks += 1
        else:
            violence_marks = 0

        if violence_marks >= 3:
            if not violent:
                print ('violence detected')
                print (result)
            violent = True
        elif (violence_marks < 3 and violent):
            print (result)
            violent = False
            clear()


    # waitKey(1) will wait 1 milisecond for the break key (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        sess.close()
        break
    current += 1


