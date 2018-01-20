from mnist import MNIST
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

# File Reading #
#-------------------------#
mndata = MNIST('data')									#Specify data directory
train_image, train_label = mndata.load_training()		#Read training image and label
test_image,  test_label  = mndata.load_testing()		#Read testing image and label

#Varaible Definition
#-------------------------#
global num_per_digit

orig_train_size = len(train_image)
test_size		= len(test_image)
sample_size 	= int(orig_train_size*0.1)
num_per_digit 	= int(sample_size/10)

#Euclidean Distance between 2 function
#-------------------------#
def distance(a,b):
	a_q = np.asarray(a,dtype=np.float32)
	b_q = np.asarray(a,dtype=np.float32)
	sum_of_sqr = (a_q-b_q)**2
	result = np.sum(sum_of_sqr)**0.5
	return result
	# sum_of_sqr= 0
	# for i in range(len(a)):
	# 	sum_of_sqr+=(a[i]-b[i])**2
	# result = sum_of_sqr**0.5
	# return result

#Merge sort sampling function
#-------------------------#
# def merge_sort(train_image):
# 	if(len(train_image)==1):
# 		return train_image
# 	else:
# 		pivot = len(train_image)/2
# 		left_side 	= merge_sort(train_image[0:pivot])
# 		right_side	= merge_sort(train_image[pivot:len(train_image)])
# 		sorted_train= []
# 		while(len(left_side)!=0 and len(right_side)!=0):

#Merge sort sampling function
#-------------------------#
def sort(train_image):
	num_image = len(train_image)
	num_pixel = len(train_image[0])
	avg_train = np.zeros(num_pixel)
	all_diff  = np.zeros((num_image,num_pixel))
	sum_diff  = np.zeros(num_image)

	for i in range(num_pixel):
		avg_train[i]=float(sum(train_image[:][i]))/float(len(train_image))

	for i in range(num_image):
		all_diff[i] = abs(train_image[i]-avg_train)

	for i in range(num_image):
		sum_diff[i] = np.sum(all_diff[i,:])
	result = [(sum_diff[i], train_image[i]) for i in range(num_image)]
	reversed(sorted(result))
	return result
	# diff_array = np.zeros((len(train_image),len(train_image)))
	# for i in range(len(train_image)):
	# 	for j in range(len(train_image)):
	# 		print(i,j)
	# 		if(j>=i):
	# 			continue
	# 		else:
	# 			dist = distance(train_image[i],train_image[j])
	# 			diff_array[i,j]= dist
	# 			diff_array[j,i]= dist
	# unique_array = []
	# for i in range(len(train_image)):
	# 	unique_array.append((np.sum(diff_array[i,:]),train_image[i]))
	# sorted_array=sorted(unique_array)
	# print(sorted_array)


#Merge sort sampling function
#-------------------------#
def merge_sampling(train_image, train_label):
	global num_per_digit
	sub_image =[]
	sub_label =[]
	distrib=0.95
	# Split the the input image into 10 digits
	cat_train_image = [[] for i in range(10)]
	for i in range(len(train_image)):
		num_group = train_label[i]
		# cat_train_image[num_group].append(train_image[i])
		cat_train_image[num_group].append(train_image[i])
	for i in range(10):
		result = sort(cat_train_image[i])
		for j in range(int(num_per_digit*distrib)):
			sub_image.append(result[j][1])
			sub_label.append(i)
		for j in range(int(num_per_digit*(1-distrib))):
			sub_image.append(result[len(result)-j-1][1])
			sub_label.append(i)
	return sub_image,sub_label
# Printing some info #
#-------------------------#
print("input size", len(train_image))
print("test size", len(test_image))
print("smaple size", sample_size)
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
new_image, new_label = merge_sampling(train_image,train_label)
print(len(new_image), len(baseline_train_image))
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

#Testing function
#-------------------------#
def test_model(train_set, train_label, test_image, test_label):
	model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree',metric='euclidean').fit(train_set)
	# dist, predict_index = model.kneighbors(test_image)
	correct=0
	for i in range(len(test_image)):
		print(i)
		# predict = onn(test_image[i],train_set, train_label)
		dist, index = model.kneighbors([test_image[i]])
		index = index[0][0]
		# index = predict_index[i]
		if(train_label[index]==test_label[i]):
			correct+=1
	rate = float(correct)/float(len(test_image))
	return rate

# base = test_model(baseline_train_image,baseline_train_label, test_image,test_label)
new  = test_model(new_image,new_label, test_image,test_label)
# print("baseline model "+str(base))
print("new model " + str(new))