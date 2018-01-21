
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

#0 is default camera for VideoCapture arg
cap = cv2.VideoCapture(0)

#How often we want to get a picture
fps = 20
current = 0

#Load tensorflow session----------------------------------------------------
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

#Get Camera feed
while(True):

    #Only captures image if it has gone through 20 loops.
    if (current % fps == 0):
        #Resets current (current is the current loop number) to 0.
        current = 0
        # Capture frame-by-frame (frame is the actual image)
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame',frame)

        result = predict.prediction(frame, sess, y_true, y_pred, x)

        if (result.item(0) > 95):
            violence_marks += 1
        else:
            violence_marks = 0

        if violence_marks >= 3:
            print ('violence')

    
    #waitKey(1) will wait 1 milisecond for the break key (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        sess.close()
        break
    current += 1


