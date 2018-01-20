import numpy as np
import tensorflow as tf
import part1

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

def getResponsibilities(pairwise_matrix, k):
	temp_array = tf.transpose(tf.reciprocal(result), perm=[2, 3, 0, 1])
	responsibilities_array = [0] * temp_array.shape[1]
	values, indices = tf.nn.top_k(temp_array, k)
	for i in range(len(indicies)):
		responsibilities_array[indicies[i]] = 1/k
	return(responsibilities_array)



testDataTensor = tf.convert_to_tensor(testData[0].astype(np.float32))
result = part1.pairwise_square_euclid_distance(testDataTensor, trainDataTensor)

training_mse = {}
validation_mse = {}
test_mse = {}
for k in [1, 3, 5, 50]:
	validation_mse[k] = 0
	for i in range(len(validData)):
		validDataTensor = tf.convert_to_tensor(validData[0].astype(np.float32))
		result = part1.pairwise_square_euclid_distance(validDataTensor, trainDataTensor)
		resp = getResponsibilities(result, k)
		guess = 0
		for j, weight in enumerate(responsibilities_array):
			guess += weight * trainTarget[j]
		validation_mse[k] += (1/len(validData)) * (guess - validTarget[i])**2

print(validation_mse)

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

