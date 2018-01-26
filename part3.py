import argparse
import numpy as np
import tensorflow as tf
import part1
import part2
import sys
import operator
#import matplotlib.pyplot as plt

sess = tf.Session()


def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
	                                 data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	                                 data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	                            target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	                            target[rnd_idx[trBatch + validBatch + 1:-1], task]
	return trainData, validData, testData, trainTarget, validTarget, testTarget

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', required=True)
parser.add_argument('--input_target', required=True)
args = parser.parse_args()

train_data, valid_data, test_data, train_tar, valid_tar, test_tar = data_segmentation(args.input_data, args.input_target, 0)

#test_data = test_data[0:5]
#test_tar = test_tar[0:5]

validation_mse = {}
predict_labels = {}
k_accuracy = {}

for k in [1, 5, 10, 25]:
	k_accuracy[k] = 0
	print("K NOW: " + str(k))
	validation_mse[k] = 0
	testResp = part2.getTotalResponsibilities(train_data, test_data, valid_data, train_tar, test_tar, valid_tar, k)
	for key, resp in testResp.items():
		for i in range(6):
			predict_labels[i] = 0
		guess = 0
		for j, weight in enumerate(resp):
			if(weight > 0):
				predict_labels[train_tar[j]] += 1
				print("shape of train_data[j]: " + str(train_data[j].shape))
		print("Predictions for image " + str(key) + " are: ")
		max_entry = max(predict_labels.items(), key=operator.itemgetter(1))
		print(str(max_entry[0]) + " with total count: " + str(max_entry[1]) + " remainder: " + str(k-max_entry[1]))
		print("Correct answer: " + str(valid_tar[key]) + "\n")
		if(max_entry[0] == valid_tar[key]):
			k_accuracy[k] += 1

print("VALIDATION K")
print(k_accuracy)
sys.exit()

k_valid = 0

for k in [k_valid]:
	print("K NOW: " + str(k))
	validation_mse[k] = 0
	testResp = part2.getTotalResponsibilities(train_data, valid_data, test_data, train_tar, valid_tar, test_tar, k)
	for key, resp in testResp.items():
		for i in range(6):
			predict_labels[i] = 0
		guess = 0
		for j, weight in enumerate(resp):
			guess += weight * train_tar[j]
			if(weight > 0):
				predict_labels[train_tar[j]] += 1
				print("shape of train_data[j]: " + str(train_data[j].shape))
		print("Predictions for image " + str(key) + " are: ")
		max_entry = max(predict_labels.items(), key=operator.itemgetter(1))
		print(str(max_entry[0]) + " with total count: " + str(max_entry[1]) + " remainder: " + str(k-max_entry[1]))
		print("Correct answer: " + str(test_tar[key]) + "\n")


		#print(predict_labels)
		#print("MSE with k: " + str(k) + " for image: " + str(key) + " is: " + str(((guess - test_tar[key])**2)))
		validation_mse[k] += (1/len(test_data)) * (guess - test_tar[key])**2
