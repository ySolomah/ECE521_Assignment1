import numpy as np
import tensorflow as tf
import part1
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

def getResponsibilities(data_matrix, target_matrix, data_valid, target_valid, pairwise_matrix, k):
	#print("Top : " + str(k))
	temp_array = tf.transpose(tf.reciprocal(pairwise_matrix), perm=[2, 3, 0, 1])
	temp_array_shape = (sess.run(temp_array)).shape
	#print("Temp array shape: " + str(temp_array_shape))
	responsibilities_array = [0] * temp_array_shape[3]
	#print("Resp array shape: " + (str(len(responsibilities_array))))
	values, indices = tf.nn.top_k(temp_array, k)
	values = sess.run(tf.reshape(values, [k]))
	indices = sess.run(tf.reshape(indices, [k]))
	#print("Shape of values: " + str(values.shape))
	#print("Shape of indices: " + str(indices.shape))
	#print("Validation data: " + str(data_valid) + " valid target: " + str(target_valid))
	for i in range(len(indices)):
		#print("Index: " + str(indices[i]) + " (casted as int): " + str(int(indices[i])) + " with dataVal: " + str(data_matrix[int(indices[i])]) + " with target: " + str(target_matrix[(indices[i])]))
		responsibilities_array[int(indices[i])] = 1/k
	return(responsibilities_array)

'''
testDataTensor = tf.convert_to_tensor(testData[0].astype(np.float32))
result = part1.pairwise_square_euclid_distance(testDataTensor, trainDataTensor)

training_mse = {}
validation_mse = {}
test_mse = {}
for k in [1, 3, 5, 50]:
	validation_mse[k] = 0
	for i in range(len(validData)):
		validDataTensor = tf.convert_to_tensor(validData[i].astype(np.float32))
		result = part1.pairwise_square_euclid_distance(validDataTensor, trainDataTensor)
		resp = getResponsibilities(trainData, trainTarget, validData[i], validTarget[i], result, k)
		guess = 0
		for j, weight in enumerate(resp):
			guess += weight * trainTarget[j]
		validation_mse[k] += (1/len(validData)) * (guess - validTarget[i])**2

print(validation_mse)
sys.exit()

#print(validDataTensor.get_shape())
print(result.shape)
k_array = sess.run(tf.transpose(tf.reciprocal(result), perm=[2, 3, 0, 1]))
print(k_array)
values, indices = tf.nn.top_k(k_array, 5)
values = sess.run(values)
indices = sess.run(indices)
print("Val:")
print(testData[0])
print(1/values)
print(indices)
for i in range(len(indices)):
	print("Train at: ")
	print(trainData[indices[i]])
	guesses.append(trainTarget[indices[i]])

print("Guesses:")
print(guesses)
print(testTarget[0])
'''

def getTotalResponsibilities(train_data, valid_data, test_data, train_tar, valid_tar, test_tar, k):
	resp_total = {}
	if(len(train_data.shape) == 3):
		train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))	
		valid_data = valid_data.reshape((valid_data.shape[0], valid_data.shape[1] * valid_data.shape[2]))
		test_data = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
	train_data_tensor = tf.convert_to_tensor(train_data.astype(np.float32))
	for i in range(len(test_data)):
		print("Processing test_data: " + str(i))
		test_data_tensor = tf.convert_to_tensor(test_data[i].astype(np.float32))
		result = part1.pairwise_square_euclid_distance(test_data_tensor, train_data_tensor)
		resp_local = getResponsibilities(train_data, train_tar, test_data[i], test_tar[i], result, k)
		resp_total[i] = resp_local
	return(resp_total)



















