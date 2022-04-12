# This file is our implementation of the Deep Artistic Style concept outlined in
# 'A Neural Algorithm of Artistic Style'. 
# Santhosh Bomminani, Jake Nickel
# CS 445, 2022

import tensorflow as tf
from tensorflow import keras
import numpy as np

def init_model():
	# The layer that contains the "content" understanding of the input images
	content_layer = "block5_conv2" 

	full_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	full_model.trainable = False
	content_output = full_model.get_layer(content_layer).output
	specific_model = keras.Model(full_model.input, content_output)
	return specific_model


	
