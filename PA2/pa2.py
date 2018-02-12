import numpy as np
import random
from matplotlib import pyplot as pplot
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

# Function to conver a given list of string features into float array
def quick_convert(datum):
	temp =[1.0]
	for feat in datum[1:]:
		temp.append(float(feat))
	return temp

# Function that calculate gradient descent of a singel feature
def gradient_descent(w,label,feature,index):
	eta=0.0025
	grad=0.0
	comb=zip(label, feature)
	for datum in comb:
		neg_prob = 1/(1+np.exp(datum[0]*np.dot(w,datum[1])))

		grad += datum[0]*datum[1][index]*neg_prob
	new_w=w
	new_w[index] = new_w[index]+eta*grad
	return new_w


# Loss function
def lost_func(label, feature, w):
	lost_sum=0.0
	comb = zip(label, feature)
	for data in comb:
		curr_label = data[0]
		curr_feat  = np.asarray(data[1])
		lost_sum+=np.log(1+np.exp(-curr_label*(np.dot(curr_feat,w)) ))
	return lost_sum


# default logistic regression using sklearn
def default_regression(label,feature):
	default_model = linear_model.LogisticRegression(C=10e8,fit_intercept=False )
	default_model.fit(feature,label)
	return default_model.coef_[0]



# Function that picks a random axis for regression at random
def random_regression(label, feature, iter):
	random.seed()
	num_feat = len(feature[0])
	w=np.zeros(num_feat)
	total_loss = []
	for i in range(iter):
		rand_feat = random.randrange(num_feat)
		total_loss.append(lost_func(label,feature,w))
		w = gradient_descent(w, label, feature, rand_feat)
	return (w, total_loss)

# Function to be worked on
def model_regression(label,feature,iter):
	all_loss = [0.0]*14
	w=np.zeros(14)
	total_loss=[]
	for i in range(iter):
		if(i%5==0):
			for j in range(14):
				w = gradient_descent(w, label, feature, j)
				all_loss[j]=lost_func(label,feature,w)
			seq = zip(all_loss, range(14))
			sorted_seq = sorted(seq,key=lambda x: x[0])
			sorted_seq.reverse
			idx_seq = [sorted_seq[i][1] for i in range(3)]
		total_loss.append(lost_func(label,feature,w))
		w = gradient_descent(w, label, feature, idx_seq[i%3])
	total_loss.append(lost_func(label,feature,w))
	# print(total_w)
	return (w, total_loss)


# Open file
file_name = "wine.txt"
raw_file = open(file_name,'r')

# Read raw data
raw_data = raw_file.readlines()
raw_data = [raw_datum.strip("\n").split(",") for raw_datum in raw_data]

# Process data
label = []
feature = np.asarray([quick_convert(datum) for datum in raw_data if int(datum[0])<3])
for datum in raw_data:
	if(int(datum[0])==1):
		label.append(-1)
	elif(int(datum[0])==2):
		label.append(1)

#Rescale feature
norm_feat = feature
for i in range(13):
	if(np.average(feature[:,i+1])>=3):
		normalizer= MinMaxScaler(feature_range=(0.,5.))
		d2_conv   = np.reshape(feature[:,i+1],(len(feature[:,i+1]),1))
		temp=normalizer.fit_transform(d2_conv)
		for j in range(len(feature[:,0])):
			norm_feat[j,i+1]=temp[j,0]

# print(feature[:,13])
# print(norm_feat[:,13])


for i in range(len(feature[0])):
	feature[:,i]

w = default_regression(label,feature)
w2, loss_1 = random_regression(label,norm_feat, 1000)
w3, loss_2 = model_regression(label,norm_feat, 1000)

pplot.plot(range(len(loss_1)),loss_1)
# pplot.plot(range(len(loss_2)),loss_2)
pplot.show()
# print(w2)
# print(w)
print("lost-default",lost_func(label,feature,w))
print("lost-random",lost_func(label,feature,w2))
print("lost-prototype",lost_func(label,feature,w3))