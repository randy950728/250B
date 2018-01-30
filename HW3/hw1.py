from mnist import MNIST
# import scipy.stats as stats
# from matplotlib import pyplot
# import random
# from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import multivariate_normal


# File Reading #
#-------------------------#
mndata = MNIST('data')									#Specify data directory
raw_image,	raw_label 	= mndata.load_training()		#Read training image and label
test_image,	test_label  = mndata.load_testing()		#Read testing image and label

#Varaible Definition
#-------------------------#
raw_size = len(raw_image)
train_size 		= 50000
valid_size		= 10000
test_size		= len(test_image)


#Split training set
#-------------------------#
train_image = raw_image[0:train_size+1]
train_label = raw_label[0:train_size+1]
valid_image = raw_image[train_size+1: raw_size]
valid_label = raw_label[train_size+1: raw_size]


#Caclculate digit prob.
#-------------------------#
num_digit=np.zeros(10)
for i in train_label:
	num_digit[i]+=1

prob_digit = num_digit/len(train_label)


# Classify each training data point
#-------------------------#
cat_digit=[]
for i in range(10):
	cat_digit.append([])

for i in range(len(train_label)):
	curr_digit=train_label[i]
	cat_digit[curr_digit].append(train_image[i])
for i in range(10):
	cat_digit[i] = np.asarray(cat_digit[i])

digit_mean=[]
digit_cov=[]
for i in range(10):
	mean = np.mean(cat_digit[i],axis=0)
	cov  = np.cov(cat_digit[i].T)
	digit_mean.append(mean)
	digit_cov.append(cov)
c=0.5
identity = np.identity(len(train_image[0]))
identity = c*identity

digit_cov = [digit_cov[i]+identity for i in range(10)]
for i in range(10):
	prob =np.log(multivariate_normal.pdf(train_image[0],digit_mean[i],digit_cov[i]))
	print(prob)
print(train_label[0])

