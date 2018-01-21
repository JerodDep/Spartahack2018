import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# First, pass the path of the image
def prediction(image, sess, y_true, y_pred, x):
    image_size=128
    num_channels=3
    images = []
    # Reading the image using OpenCV
    #image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    y_test_images = np.zeros((1, 2))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [violent non-violent]
    print(result)
