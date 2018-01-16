import tensorflow as tf

def pairwise_square_euclid_distance(A, B):
	A_exp = tf.expand_dims(A, 0)
	B_exp = tf.expand_dims(B, 1)
	result = A_exp - B_exp
	result2 = tf.square(result)
	result3 = tf.expand_dims(result2, 0)
	sess = tf.Session()
	tensor_shape = result3.get_shape()
	print(tensor_shape)
	print("BEGIN SUBTRACT")
	print(sess.run(result))
	print("END SUBTRACT")
	print("BEGIN SQUARE")
	print(sess.run(result3))
	print("END SQUARE")
	in_channels = tensor_shape[3]
	kernel = tf.fill([1, 1, in_channels, 1], 1.0)
	result4 = tf.nn.conv2d(result3, kernel, strides=[1, 1, 1, 1], padding='VALID')
	final_result = sess.run(result4)
	print("BEGIN SUM")
	print(final_result)
	print("END SUM")
	return(final_result)
