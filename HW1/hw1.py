from mnist import MNIST
import random

# File Reading #
#-------------------------#
mndata = MNIST('data')									#Specify data directory
train_image, train_label = mndata.load_training()		#Read training image and label
test_image,  test_label  = mndata.load_testing()		#Read testing image and label

#Varaible Definition
#-------------------------#
orig_train_size = len(train_image)
test_size		= len(test_image)
sample_size 	= int(orig_train_size*0.8)

# Printing some info #
#-------------------------#
print("input size", len(train_image))
print("test size", len(test_image))
print("smaple size", len(sample_size))
# print(mndata.display(train_image[1]))

# Baseline sampling
#-------------------------#
index_array = range(0, orig_train_size)
baseline_train_image = []
baseline_train_label = []
curr_size=0
while(curr_size<sample_size):
	index_index = random.randrange(len(index_array))
	index = index_array[index_index]
	baseline_train_image.append(train_image[index])
	baseline_train_label.append(train_label[index])
	del(index_array[index_index])
	curr_size+=1

print(len(baseline_train_image))

# Data sampling
#-------------------------#
index_array = range(0,orig_train_size)


#Euclidean Distance between 2 function
#-------------------------#
def distance(a,b):
	sum_of_sqr=0
	for i in range(len(a)):
		sum_of_sqr+=(a[i]-b[i])**2
	result = sum_of_sqr**0.5
	return result

#1-NN function
#-------------------------#
def onn(item, train_image, train_label):
	nn_val=1000000
	nn_label=-1
	for i in range(len(train_image)):
		dist = distance(item,train_image[i])
		if(dist<nn_val):
			nn_val = dist
			nn_label = train_label[i]

	return nn_label

