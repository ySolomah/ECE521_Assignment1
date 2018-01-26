import numpy as np
import tensorflow as tf
import sys

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
         + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

trainDataTensor = tf.convert_to_tensor(trainData.astype(np.float32))

sess = tf.Session()

guesses = []

def pairwise_square_euclid_distance(A, B):
	A_exp = tf.expand_dims(A, 0)
	B_exp = tf.expand_dims(B, 1)
	result = A_exp - B_exp
	result2 = tf.square(result)
	result3 = tf.expand_dims(result2, 0)
	sess = tf.Session()
	tensor_shape = result3.get_shape()
	in_channels = tensor_shape[3]
	kernel = tf.fill([1, 1, in_channels, 1], 1.0)
	result4 = tf.nn.conv2d(result3, kernel, strides=[1, 1, 1, 1], padding='VALID')
	final_result = sess.run(result4)
	return(final_result)


def getResponsibilities(trainData, validData, k):
	temp_array = pairwise_square_euclid_distance(trainData, validData)
	tensor_shape = temp_array.shape
	temp_array = tf.reciprocal(tf.convert_to_tensor(temp_array.astype(np.float32)))
	temp_array = tf.reshape(temp_array, [tensor_shape[1], tensor_shape[2]])
	values, indices = tf.nn.top_k(temp_array, k)
	indices = sess.run(indices)
	values = sess.run(values)
	return(indices, values)



mse_track = {}
for k in [1, 3, 5, 50]:
	mse_track[k] = 0
	validDataTensor = tf.convert_to_tensor(validData.astype(np.float32))
	indices, values = getResponsibilities(trainDataTensor, validDataTensor, k)
	tensor_weights = tf.one_hot(indices, depth=len(trainTarget), on_value=1/k, off_value=0.0, dtype=tf.float32)
	tensor_weights = tf.reduce_sum(tensor_weights, axis=1)
	train_tar = tf.transpose(tf.convert_to_tensor(trainTarget), perm=[1, 0])
	test_exp = tf.constant(0.0, shape=[len(validData)], dtype=tf.float32)
	test_exp = tf.expand_dims(test_exp, axis=1)
	result = tf.add(tf.to_float(train_tar), tf.to_float(test_exp))
	predict = tf.multiply(result, tensor_weights)
	print("GUESS: " + str(k))
	final_result = tf.reduce_sum(predict, axis=1)
	final_tar = tf.reshape(tf.convert_to_tensor(validTarget.astype(np.float32)), [len(validTarget)])
	mse = 1/(2 * len(testTarget)) * sess.run(tf.reduce_sum(tf.square(tf.subtract(final_result, final_tar))))
	print(mse)


sys.exit()

