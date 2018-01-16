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


testDataTensor = tf.convert_to_tensor(testData[0].astype(np.float32))
result = part1.pairwise_square_euclid_distance(testDataTensor, trainDataTensor)
#print(validDataTensor.get_shape())
print(result.shape)
k_array = sess.run(tf.transpose(tf.reciprocal(result), perm=[2, 3, 0, 1]))
print(k_array)
values, indices = tf.nn.top_k(k_array, 3)
values = sess.run(values)
indices = sess.run(indices)
print(values)
print(indices)
for i in range(len(indices)):
	guesses.append(trainTarget[indices[i]])

print(guesses)
print(testTarget[0])

