
# coding: utf-8

# ### Reference:
# how to use Mnist dataset:<br>
# https://rasbt.github.io/mlxtend/user_guide/data/mnist_data/#example-1-dataset-overview

# In[10]:


import sys
import random
import numpy as np
import scipy as sp
import scipy.stats
import mlxtend.data
import time

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


# In[11]:


MNIST_PATH = "data/";


# In[12]:


train_images = MNIST_PATH + "train-images-idx3-ubyte";
train_labels = MNIST_PATH + "train-labels-idx1-ubyte";
test_images = MNIST_PATH + "t10k-images-idx3-ubyte";
test_labels = MNIST_PATH + "t10k-labels-idx1-ubyte";


# In[13]:


train_data = mlxtend.data.loadlocal_mnist(train_images, train_labels);
test_data = mlxtend.data.loadlocal_mnist(test_images, test_labels);


# In[14]:


# build kd-tree for input data
def build_kdtree(data, leaf_size):
    return KDTree(data[0], leaf_size)


# In[15]:


def random_prototype_selector(dataset, M):
    images, labels = dataset[0], dataset[1]
    pimages, plabels = [], []

    # use shuffle method
    slist = list(zip(images, labels))
    random.shuffle(slist)
    images, labels = zip(*slist) # unzip list

    return images[:M], labels[:M]

    # use probability method
#     N = len(images)
#     p = float(M) / N   # probability to keep a sample in subset
#     for (image, label) in zip(images, labels):
#         if random.random() <= p:
#             pimages.append(image)
#             plabels.append(label)
#         if len(pimages) == M:
#             break
#     return pimages, plabels


# In[41]:


def kmeans_prototype_selector(dataset, M, n_clusters):
    images, labels = dataset[0], dataset[1]
    pimages, plabels = [], []

    slist = list(zip(images, labels))
    random.shuffle(slist)
    images, labels = zip(*slist) # unzip list

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(images)

    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    c_ratio_dict = dict(zip(unique, counts))
    for key, value in c_ratio_dict.items():
        c_ratio_dict[key] = M * float(value) / len(images)

    print(c_ratio_dict)
    for (image, label, c) in zip(images, labels, kmeans.labels_):
        if c_ratio_dict[c] > 0:
            pimages.append(image)
            plabels.append(label)
            c_ratio_dict[c] -= 1
        if len(pimages) == M:
            break;
    return pimages, plabels;


# In[17]:


# def kmeans_prototype_selector(dataset, M, n_clusters):
#     images, labels = dataset[0], dataset[1]
#     pimages, plabels = [], []

#     slist = list(zip(images, labels))
#     random.shuffle(slist)
#     images, labels = zip(*slist) # unzip list

#     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(images)

#     unique, counts = np.unique(kmeans.labels_, return_counts=True)
#     c_ratio_dict = dict(zip(unique, counts))
#     for key, value in c_ratio_dict.items():
#         c_ratio_dict[key] = round(M * float(value) / len(images))

#     for i in range(n_clusters):
#         distance_to_cluster = kmeans.transform(images)[:, i]
#         sample_num_of_cluster = c_ratio_dict[i]
#         idx_of_pts = np.argsort(distance_to_cluster)[::][:sample_num_of_cluster]
#         for idx in idx_of_pts:
#             pimages.append(images[idx])
#             plabels.append(labels[idx])
# #             print(idx)
# #         print(plabels)
#     return pimages, plabels;


# In[18]:


def cal_accuracy(tree, prototype, test_data):
    correct = 0
    for (timage, tlabel) in zip(test_data[0], test_data[1]):
        dist, ind = tree.query([timage], k=1)
        if prototype[1][ind[0][0]] == tlabel:
            correct += 1
    return float(correct) / len(test_data[0])


# In[46]:


def cal_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


# In[47]:


def run_random_model(train_data, test_data, sample_size, leaf_size=10, turns=10, confidence=0.95, log=True):
    avg_accuracy = 0
    acc_list = []

    for i in range(turns):
        prototype = random_prototype_selector(train_data, sample_size)
        if log: print("Turns:", i, "/ Size of prototype:", len(prototype[0]))
        tree = build_kdtree(prototype, leaf_size)
        acc = cal_accuracy(tree, prototype, test_data)
        acc_list.append(acc)
        avg_accuracy += acc
    print("Average Accuracy:", float(avg_accuracy) / turns)
    if turns > 1: print("Standard Deviation", np.std(acc_list))
    if turns > 1: print("Confidential Interval", cal_confidence_interval(acc_list))


# In[48]:


def run_kmeans_model(train_data, test_data, sample_size, cluster_size=10, leaf_size=10, turns=10, confidence=0.95, log=True):
    avg_accuracy = 0
    acc_list = []

    for i in range(turns):
        prototype = kmeans_prototype_selector(train_data, sample_size, cluster_size)
        if log: print("Turns:", i, "/ Size of prototype:", len(prototype[0]))
        tree = build_kdtree(prototype, leaf_size)
        acc = cal_accuracy(tree, prototype, test_data)
        acc_list.append(acc)
        avg_accuracy += acc
    print("Average Accuracy:", float(avg_accuracy) / turns)
    if turns > 1: print("Standard Deviation", np.std(acc_list))
    if turns > 1: print("Confidential Interval", cal_confidence_interval(acc_list))




start_time = time.time()
run_random_model(train_data, test_data, 10000, turns=30)
print("Running time", time.time() - start_time)

# In[ ]:


start_time = time.time()
run_kmeans_model(train_data, test_data, 10000, turns=30)
print("Running time", time.time() - start_time)

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