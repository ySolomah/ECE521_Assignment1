import tensorflow as tf

def pairwise_square_euclid_distance(A, B):
	A_exp = tf.expand_dims(A, 0)
	B_exp = tf.expand_dims(B, 1)
	result =