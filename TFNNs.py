import numpy as np
import tensorflow as tf


#Features is input data
def build_cnn(features): 
	#Reshape dim depends on dim on atari image
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	conv1 = tf.layers.conv2d(inputs = input_layey
		, filters = 32
		, kernel_size=[5,5]
		, padding = "same"
		, activation = tf.nn.relu)

	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)



batch_size = 100
image_width = 10
image_height = 10
image_size = image_width*image_height

action_number = 5

hidden1_units = 100
hidden2_units = 100
def build_tfnn():
	input_placeholder = tf.placeholder(tf.float32, shape(batch_size, image_size))

	action_placeholder = tf.placeholder(tf.float32, shape(batch_size, action_number))

	#use name_scope to make naming easy
	with tf.name_scope("hidden1"):
		weights = tf.Variable(tf.truncated_normal([image_size, hidden1_units])
			, stddev = 1.0/np.sqrt(image_size)
			, name = "weights")
		#first input is dimension + init type
		#second input is init parameters
		#third is name. This has name hidden1/weights

		biases = tf.Variable(tf.zeros([hidden1_units])
			, name = "biases")


		hidden1 = tf.nn.relu(tf.matmul(input_placeholder, weights) + biases)

	with tf.name_scope("hidden2"):			
		weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units])
			, stddev = 1.0/np.sqrt(image_size)
			, name = "weights")
		#first input is dimension + init type
		#second input is init parameters
		#third is name. This has name hidden1/weights

		biases = tf.Variable(tf.zeros([hidden2_units])
			, name = "biases")


		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)


	with tf.name_scope("output_layer"):
		weights = tf.Variable(tf.truncated_normal([hidden2_units, action_number])
			, stddev = 1.0/np.sqrt(image_size)
			, name = "weights")
		#first input is dimension + init type
		#second input is init parameters
		#third is name. This has name hidden1/weights

		biases = tf.Variable(tf.zeros([action_number])
			, name = "biases")
		output = tf.nn.tanh(tf.matmul(hidden2, weights) + biases)


	



###https://www.tensorflow.org/versions/r0.12/tutorials/mnist/tf/

###https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py


###https://www.tensorflow.org/tutorials/layers








