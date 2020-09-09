#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimenting with pretrained tensorflow NNs

Created on Wed Sep  9 14:29:35 2020

@author: riversdale
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4
m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4")
])
m.build([None, 128, 128, 3])  # Batch input shape.

# img is a variable of my own Image class
img_tensor = tf.convert_to_tensor(img.data)
# must have shape [batch_size, height, width, 3]
# so we need to insert the singleton "batch" dimension"
img_tensor = tf.expand_dims(img_tensor, axis=0)

# Run the model
logits = m(img_tensor) # run th model
argmax = tf.argmax(logits[0]) # predicted class of image
softmax = tf.nn.softmax(logits[0]) # probabilities of classes

# load the class names
# downloaded from here:https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
f = open("ImageNetLabels.txt", "r")
classes = np.array([line.strip() for line in f.readlines()])
f.close()

#sort the probabilities
probs = np.sort(softmax)
sortix = np.argsort(softmax)
# NB these will be in ascending order. We want to reverse and get top 10
probs = probs[:-10:-1]
sortix = sortix[:-10:-1]
labels = classes[sortix] # names of the top 10 most likely classes

# trying classification on a known image (and doing multiple at once)
image = tf.keras.preprocessing.image.load_img('mallard_rgb_128.png')
input_arr = tf.keras.preprocessing.image.img_to_array(image)
img_tensor = tf.stack([tf.convert_to_tensor(img.data), input_arr/255])

# yep seems to work (98% probability it is a "drake")

