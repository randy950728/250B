from mnist import MNIST
import numpy as np
import random
from scipy.stats import multivariate_normal
from matplotlib  import pyplot


# File Reading #
#-------------------------#
mndata = MNIST('data')									#Specify data directory
raw_image,	raw_label 	= mndata.load_training()		#Read training image and label
test_image,	test_label  = mndata.load_testing()			#Read testing image and label

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
cat_digit=[[] for i in range(10)]
for i in range(len(train_label)):
	curr_digit=train_label[i]
	cat_digit[curr_digit].append(train_image[i])


# Convert training data of each digit into numpy array
#-------------------------#
cat_digit = [np.asarray(cat_digit[i]) for i in range(10)]


# Calculate mean and covaraince of each digit
#-------------------------#
digit_mean=[]
digit_covv=[]
for i in range(10):
	mean = np.mean(cat_digit[i],axis=0)
	cov  = np.cov(cat_digit[i].T)
	digit_mean.append(mean)
	digit_covv.append(cov)


# smoothing covariance matrix
#-------------------------#
# C=[5062,5092,5125,5187]
C=range(2000,4000,50)
print(C)
# c=5100
c=3350
result =[]

ci= c*np.identity(len(train_image[0]))
digit_cov = [digit_covv[i]+ci for i in range(10)]


# Classify each training data point
#-------------------------#
digit_norm=[]
for i in range(10):
	digit_norm.append( multivariate_normal(digit_mean[i],digit_cov[i]))


# Classify each training data point
#-------------------------#
correct=0.
for i in range(len(valid_label)):
	temp = np.zeros(10)
	for j in range(10):
		temp[j] = digit_norm[j].logpdf(valid_image[i]) + np.log(prob_digit[j])

	best_guess = np.argmax(temp)
	# print(temp)
	# print(valid_label[i],best_guess)
	if(best_guess==valid_label[i]):
		correct+=1

	if(i%1000==0):
		print(i)


# Classify each training data point
#-------------------------#
acc = correct/len(valid_label)
print("C="+str(c)+"  error:"+str(1-acc))
# result.append(1-acc)

# pyplot.plot(C,result)
# pyplot.title("Validation set error rate VS. covariance smoothing constant-c")
# pyplot.xlabel("smoothing constant-c")
# pyplot.ylabel("Validation set error rate")
# pyplot.show()
# quick = np.asarray(result)
# print("argmax",np.argmax(result))

# Test on testing set
#-------------------------#
correct=0.
for i in range(len(test_label)):
	temp = np.zeros(10)
	for j in range(10):
		temp[j] = digit_norm[j].logpdf(test_image[i]) + np.log(prob_digit[j])

	best_guess = np.argmax(temp)
	# print(temp)
	# print(valid_label[i],best_guess)
	if(best_guess==test_label[i]):
		correct+=1

	if(i%1000==0):
		print(i)
# Classify each training data point
#-------------------------#
acc = correct/len(valid_label)
print("Test result "+"C="+str(c)+"  error:"+str(1-acc))