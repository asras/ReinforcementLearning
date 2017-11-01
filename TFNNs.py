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


	
class SimpleQLearner:

	def __init__(self):
		self._build_model()

	

	def _build_model(self):
		hidden1_units = 100
		hidden2_units = 100
		image_size = 9*9*3
		action_number = 83
		self.input_placeholder = tf.placeholder(tf.float32, shape(batch_size, image_size))

		self.y_pl = tf.placeholder(shape = [None], dtype=tf.float32, name="y")

		#We put the chosen actions in here so we can gather the correct outputs
		#for the loss function
		self.action_placeholder = tf.placeholder(tf.float32, shape(batch_size, action_number))

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
				, stddev = 1.0/np.sqrt(hidden1_units)
				, name = "weights")
			#first input is dimension + init type
			#second input is init parameters
			#third is name. This has name hidden1/weights

			biases = tf.Variable(tf.zeros([hidden2_units])
				, name = "biases")


			hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)


		with tf.name_scope("output_layer"):
			weights = tf.Variable(tf.truncated_normal([hidden2_units, action_number])
				, stddev = 1.0/np.sqrt(hidden2_units)
				, name = "weights")
			#first input is dimension + init type
			#second input is init parameters
			#third is name. This has name hidden1/weights

			biases = tf.Variable(tf.zeros([action_number])
				, name = "biases")
			output = tf.matmul(hidden2, weights) + biases

		self.predictions = output


		##Set up loss
		gather_indices = self.actions_placeholder
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

		self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
		self.loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.famework
		.get_global_step())

	
	def predict(self, sess, board):
		return sess.run(self.predictions, {self.input_placeholder : board})

	def update(self, sess, board, action, target_value):

		feed_dict = {self.input_placeholder : board, self.target_value:target_value
		, self.action_placeholder : action}
		global_step, _, loss = sess.run(
			[tf.contrib.framework.get_global_step(), self.train_op, self.loss]
			, feed_dict
		)
		return loss






###https://www.tensorflow.org/versions/r0.12/tutorials/mnist/tf/

###https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py


###https://www.tensorflow.org/tutorials/layers








