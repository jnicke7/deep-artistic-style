# This file is our implementation of the Deep Artistic Style concept outlined in
# 'A Neural Algorithm of Artistic Style'. 
# Santhosh Bomminani, Jake Nickel
# CS 445, 2022

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.utils import Progbar

#tf.enable_eager_execution()

def init_model(layers):
	full_model = keras.applications.VGG19(include_top=False, weights='imagenet', pooling='avg')
	full_model.trainable = False
	outputs = [full_model.get_layer(layer_name).output for layer_name in layers]
	specific_model = keras.Model(full_model.input, outputs)
	return specific_model

def gram(input_tensor):
	# Input tensor is a BxHxWxF matrix, where B is the batch, H and W are the output size,
	# and F is the number of feature maps. 
	# The i-th row and j-th column of the Gram matrix is calculated
	# by taking the inner product of the i-th feature map with the 
	# j-th feature map. 

	gram_matrix = tf.einsum('buvi,buvj->bij', input_tensor, input_tensor)
	# Now divide this result by the total number of elements in each feature map
	num_elems = input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3]
	return gram_matrix / (2.0 * tf.cast(num_elems,tf.float32))

def compute_loss(content_input, style_input, content_target, style_target):
	assert len(style_input) == len(style_target)
	content_loss = 0.5*tf.math.reduce_sum(tf.math.square(content_input - content_target))
	style_loss = 0
	for i in range(len(style_input)):
		style_loss += 1/5*tf.math.reduce_sum(tf.math.square(style_input[i] - style_target[i]))
	# Weightings for content and style loss
	alpha = 1e-3
	beta = 1
	return content_loss*alpha + style_loss*beta, content_loss, style_loss


content_img = (img_to_array(load_img('inputs/cabin.jpg')) * 1.0)[np.newaxis, :]
content_img = keras.applications.vgg19.preprocess_input(content_img)

style_img = (img_to_array(load_img('inputs/house.jpg')) * 1.0)[np.newaxis, :]
style_img = keras.applications.vgg19.preprocess_input(style_img)
# The layer that contains the "content" understanding of the input images
content_layers = ["block5_conv2"] 

# The layers that contain the "style" understanding of the input images
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

content_model = init_model(content_layers)

style_model = init_model(style_layers)

# Our models are defined, now generate the content and style targets

content_target = content_model(content_img)

style_target = style_model(style_img)

for i in range(len(style_target)):
	style_target[i] = gram(style_target[i])

result_var = tf.Variable(content_img)

optimizer = keras.optimizers.Adam(learning_rate=0.09, beta_1=0.99, epsilon=1e-1)

def gradient_descent_step(img_var):
	with tf.GradientTape() as tape:
		content_output = content_model(img_var)
		style_output = [gram(i) for i in style_model(img_var)]
		loss, content_loss, style_loss = compute_loss(content_output, style_output, content_target, style_target)
	gradient = tape.gradient(loss, img_var)
	optimizer.apply_gradients([(gradient, img_var)])
	img_var.assign(tf.clip_by_value(img_var, clip_value_min=0.0, clip_value_max = 255.0))
	return content_loss, style_loss

n_iter = 1500
pbar = Progbar(n_iter, stateful_metrics=["content_loss", "style_loss"])
for i in range(n_iter):
	content_loss, style_loss = gradient_descent_step(result_var)
	pbar.update(i+1, values=[("content_loss", content_loss), ("style_loss", style_loss)])

save_img('outputs/output.jpg', array_to_img(result_var.numpy()[0][:, :, [2, 1, 0]]))


