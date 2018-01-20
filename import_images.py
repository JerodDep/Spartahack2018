#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 00:42:57 2018

@author: Jeff Valentic

This program pulls images from the paths "mypath_V"
and "mypath_NV" and converts them to the usable format for tensorflow
it then puts them in a list of violent and non violent images.
"""

from os import listdir
from os.path import isfile, join
import tensorflow as tf

#Change mypath_V to be the path to the directory that holds the violent images.
mypath_V = "/Users/Jeff/Desktop/SpartaHack/Image Training/Violent"
#Change mypath_NV to be the path to the directory that holds the non-violent images.
mypath_NV = "/Users/Jeff/Desktop/SpartaHack/Image Training/Non-Violent"

#Importing the images and saving to the two lists violent_images and nv_images
violent_images = [f for f in listdir(mypath_V) if isfile(join(mypath_V, f))]
nv_images = [f for f in listdir(mypath_NV) if isfile(join(mypath_NV, f))]

print(violent_images)
print(nv_images)

v_que = tf.train.string_input_producer(violent_images)
nv_que = tf.train.string_input_producer(nv_images)

#VIOLENT
#read, decode and resize images
reader = tf.WholeFileReader()
filename_v, content_v = reader.read(v_que)
image = tf.image.decode_jpeg(content_v, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])

#Batching
image_batch_v = tf.train.batch([resized_image], batch_size=8)


#NON-VIOLENT
#read, decode and resize images
reader = tf.WholeFileReader()
filename_nv, content_nv = reader.read(nv_que)
image = tf.image.decode_jpeg(content_nv, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])

#Batching
image_batch_nv = tf.train.batch([resized_image], batch_size=8)

sess = tf.Session()

sess.run(image_batch_v)