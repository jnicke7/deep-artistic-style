# This file is our implementation of the Deep Artistic Style concept outlined in
# 'A Neural Algorithm of Artistic Style'. 
# Santhosh Bomminani, Jake Nickel
# CS 445, 2022

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

def init_model(layers):
	full_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	full_model.trainable = False
	outputs = [full_model.get_layer(layer_name).output for layer_name in layers]
	specific_model = keras.Model(full_model.input, outputs)
	return specific_model

def gram(input_tensor):
	pass		
	
img = (img_to_array(load_img('inputs/starry_night.jpg')) * 1.0)[np.newaxis, :]
print(np.max(img))
# The layer that contains the "content" understanding of the input images
content_layers = ["block5_conv2"] 

# The layers that contain the "style" understanding of the input images
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

content_model = init_model(content_layers)
content_output = content_model(img)
print(content_output.shape)
