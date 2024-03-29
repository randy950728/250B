from mnist import MNIST
import scipy.stats as stats
from matplotlib import pyplot
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
sample_size 	= int(7500)
num_per_digit 	= int(sample_size/10)


#Euclidean Distance between 2 function
#-------------------------#
def distance(a,b):
	a_q = np.asarray(a,dtype=np.float32)
	b_q = np.asarray(a,dtype=np.float32)
	sum_of_sqr = (a_q-b_q)**2
	result = np.sum(sum_of_sqr)**0.5
	return result


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
		all_diff[i] = (train_image[i]-avg_train)**2

	for i in range(num_image):
		sum_diff[i] = np.sum(all_diff[i,:])**0.5
	result = [(sum_diff[i], train_image[i]) for i in range(num_image)]
	result.sort(key=lambda x: x[0])
	return result


#function that finds the closest index given the value in a sorted array
#-------------------------#
def find_index(val, array):
	idx = (np.abs(array-val)).argmin()
	return idx


#Merge sort sampling function
#-------------------------#
def merge_sampling(train_image, train_label):
	global num_per_digit
	sub_image =[]
	sub_label =[]
	distrib=0.30
	# normal_distrib = [0.3415, 0.1359, 0.0214, 0.0013]
	normal_distrib = [0.3415, 0.1359, 0.0214+0.0013]

	# Split the the input image into 10 digits
	cat_train_image = [[] for i in range(10)]
	for i in range(len(train_image)):
		num_group = train_label[i]
		cat_train_image[num_group].append(train_image[i])

	for i in range(10):
		# Aquire the training digit sorted by eclidean dist.
		result = sort(cat_train_image[i])
		res_size = len(result)

		# Extract only ranking info
		diff_inf = [result[j][0] for j in range(res_size)]
		avg_diff = np.average(diff_inf)
		std_diff = np.std(diff_inf)

		# locate STD boundary
		pos_std_idx = [find_index(avg_diff+j*std_diff, diff_inf) for j in range(0,4)]
		neg_std_idx = [find_index(avg_diff-j*std_diff, diff_inf) for j in range(0,4)]
		neg_std_idx[0]=neg_std_idx[0]-1 # offset by 1 to avoid repeat sampling

		# Calculate number of samples per deviation
		samp_per_std = [int(normal_distrib[j]*num_per_digit) for j in range(3)]

		# Calculate sample spacing
		space_a = [int((pos_std_idx[j+1]-pos_std_idx[j])/samp_per_std[j]) for j in range(3)]
		space_b = [int((neg_std_idx[j]-neg_std_idx[j+1])/samp_per_std[j]) for j in range(3)]

		# Take samples from each deviation
		for j in range(3):
			curr_idx = pos_std_idx[j]
			count=0
			while(count<= samp_per_std[j]):
				sub_image.append(result[curr_idx][1])
				sub_label.append(i)
				curr_idx+=space_a[j]
				count+=1

		for j in range(3):
			curr_idx = neg_std_idx[j]
			count=0
			while(count<= samp_per_std[j]):
				sub_image.append(result[curr_idx][1])
				sub_label.append(i)
				curr_idx-=space_b[j]
				count+=1

		# fit = stats.norm.pdf(test, np.mean(test), np.std(test))
		# pyplot.hist(diff_inf,normed=True)
		# pyplot.show()

		# Print Some statis
		# print("samp_per_std", samp_per_std)
		# print(diff_inf)
		# print("avg_diff", avg_diff)
		# print("std_diff", std_diff)
		# print("pos_std_idx", pos_std_idx)
		# print("neg_std_idx", neg_std_idx)
		# print(diff_inf[pos_std_idx[0]])
		# print("space_a", space_a)
		# print("space_b", space_b)
		# print(" ")
		# # print("test",i,test
	return sub_image,sub_label


# Printing some info #
#-------------------------#
print("input size", len(train_image))
print("test size", len(test_image))
print("smaple size", sample_size)


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
		if(i%1000==0):
			print(i)
		# predict = onn(test_image[i],train_set, train_label)
		dist, index = model.kneighbors([test_image[i]])
		index = index[0][0]
		# index = predict_index[i]
		if(train_label[index]==test_label[i]):
			correct+=1
	rate = float(correct)/float(len(test_image))
	return rate

for i in range(11):
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


	# Data sampling
	#-------------------------#
	index_array = range(0,orig_train_size)
	new_image, new_label = merge_sampling(train_image, train_label)
	print(len(new_image), len(baseline_train_image))

	# Model Testing
	#-------------------------#
	base = test_model(baseline_train_image,baseline_train_label, test_image,test_label)
	# new  = test_model(new_image,new_label, test_image,test_label)
	print("baseline model "+str(base))
	# print("new model " + str(new))


# Scrapped code
#-------------------------------------------------------------#
# for j in range(int(num_per_digit*distrib)):
# 	sub_image.append(result[res_size-1-j][1])
# 	sub_label.append(i)

# for j in range(int(num_per_digit*(1-distrib))):
# 	if(res_size/2)>int(num_per_digit*(1-distrib)):
# 		sub_image.append(result[res_size/2-j][1])
# 	else:
# 		offset = num_per_digit*(1-distrib)-len(res_size)/2
# 		sub_image.append(result[res_size/2+offset-j][1])
# 	sub_label.append(i)


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