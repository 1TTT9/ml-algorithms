
# coding: utf-8

# In[1]:

"""
 ref: http://blog.cambridgecoding.com/2016/01/16/machine-learning-under-the-hood-writing-your-own-k-nearest-neighbour-algorithm/
 1. calculate dist. between any two points
 2. find the nearest neighbors base on their pairwise distance
 3. majority vote on a class labels based on nearest neighbors list
"""


# In[3]:

import random, math

def cross_validation(data, target, testSize=0.4):
    """
    parameters -
        data: raw data 
        target: outcome of data
        testSize: ratio of test set on raw data
    return - 
        trainSet: set of data-taget pairs for train
        testSet:  set of data-target pairs for test
    """
    # ref: http://stackoverflow.com/questions/6482889/get-random-sample-from-list-while-maintaining-ordering-of-items
    test_indices = random.sample(xrange(len(data)), int(len(data)*testSize))
    test_indices = sorted(test_indices)  
    # train_indices = filter(lambda x: x not in xrange(len(data)), test_indices)
    train_indices = list(set([i for i in xrange(len(data))]) - set(test_indices))

    trainSet = [(data[i], target[i]) for i in train_indices]
    testSet  = [(data[i], target[i]) for i in test_indices]
    return trainSet, testSet


# In[12]:

import numpy as np
numOfDataSet = 10
raw_data = [ (np.random.uniform(0,1,1000), np.random.uniform(0,1,1000), np.random.uniform(0,1,1000))  for i in range(numOfDataSet)]
raw_target = [np.random.uniform(0,1,1000) for i in range(numOfDataSet)]


# In[13]:

trainSet, testSet = cross_validation(raw_data, raw_target)


# In[4]:

def findPairDistance(x, y):
	"""
		consider only Euclidean distance
	"""
	dimPairs = zip(x,y)
	return math.sqrt(sum([ math.pow(p-q, 2) for p, q in dimPairs ]))


# In[5]:

def getKNearestNeighbors(train_set, test_instance, k):
    
    # step 1, calculate all distances with test_instance
    distances = [findPairDistance(train_instance, test_instance) for train_instance, train_y in train_set]
    #step 2, sorted the distances, we collected the indice set here
    #ref http://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
    
    """
        note: by sorting all values we can easily acquire the k smallest, however, it's costly and 
        time complexity O(nlogn); it may be faster to find the minimum instead, by using array, which means, 
        with space complexity O(n) and time complexity O(n)
    """    
    #step 3, pick up k the most smallest, i.e., closet instances in train_set and return
    kLargestIndice = findKLargestSet_Sort(distances, k)
    return [train_set[i] for i in kLargestIndice]

def findKLargestSet_Sort(arr, k):
    sortedIndice = sorted(xrange(len(arr)), key=lambda x: arr[x])
    return sortedIndice[:k]
    
def findKLargestSet_Randomized_Select(arr, k):
    """
        suppose 0 < k <= size of Arr 
    """
    arrIndice = [i for i in xrange(len(arr))]    
    isSorted = True
    for i in xrange(1, len(arr)):
        if arr[i-1] > arr[i]:
            isSorted = False
            break
    if isSorted:
        return arrIndice[:k+1]
    
    def __randomized_select(arr, p, r, i, arrIndice):
        q = partition(arr, p, r, arrIndice)
        k = q-p+1
        if k == i:
            return arrIndice[:k]
        elif k < i:
            return __randomized_select(arr, q+1, r, i-k, arrIndice)
        else:
            return __randomized_select(arr, p, q-1, i, arrIndice)        
    
    return __randomized_select(arr, 0, len(arr), arrIndice)


def swap(A, p, q):
    temp = A[p];A[p] = A[q];A[q] = temp

def partition(A, p, r, indiceOfA):
    if p == r:
        return p
    pivot = A[r-1]        
    m = p - 1 
    for n in xrange(p, r):
        if A[n] <= pivot:
            m += 1
            swap(A, m, n)
            swap(indiceOfA, m, n)
    swap(A, m+1, r)
    swap(indiceOfA, m+1, n)    
    return m+1


# In[6]:

from collections import Counter
def get_majority_vote(neighbors):
    # equal to trainInstance_y
    classes = [neighbor[1] for neighbor in neighbors]
    count = Counter(classes)
    return count.most_common()[0][0]


# In[7]:

"""
  Here we test our implemented algorithm whin sklearn's example
"""
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris
def main():

    #load raw data
    iris = load_iris()    
    #create train and test sets
    trainSet, testSet = cross_validation(iris.data, iris.target)
    testSet_y = zip(*testSet)[1]
    #prediction set
    predictions = []
    #arbitarily set k = 5
    k = 5
    for i in xrange(len(testSet)):
#         print "classifying test-(%d)" % i
        neighbors = getKNearestNeighbors(trainSet, testSet[i][0], k=5)
        majority_vote = get_majority_vote(neighbors)        
        predictions.append(majority_vote)
#         print "predicted label: %d, actual label: %d" % (majority_vote, testSet[i][1])
    
    #summarize performance of the classfication
    print "\nOverall accuracy of the model: %f" % accuracy_score(testSet_y, predictions)
    print "\nClassification Report:\n\n", classification_report(testSet_y, predictions, target_names = "iris-data") 
    
    
main()


# In[1]:

get_ipython().magic(u'matplotlib inline')
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import load_iris
import numpy as np 
import matplotlib.pylab as plt
iris = load_iris()
pca = RandomizedPCA(n_components = 2)
x_pca = pca.fit_transform(iris.data)
print x_pca.shape

from itertools import cycle
colors = ['b','g','r', 'c']
markers = ['+', 'o', '^', 'v']
for i,c,m in zip(np.unique(iris.target), cycle(colors), cycle(markers)):
    plt.scatter(x_pca[iris.target==i, 0], x_pca[iris.target==i, 1], c=c, marker=m, label=i, alpha=0.5)

plt.title("sklearn.dataset=iris")
plt.legend(loc="best")
plt.show()


# In[14]:

get_ipython().magic(u'matplotlib inline')
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 2)
x_svd = svd.fit_transform(zip(*trainSet)[0])

plt.scatter(x_svd[:,0], x_svd[:,1], c=zip(*trainSet)[1], s = 50, cmap=plt.cm.Paired)
plt.colorbar()
plt.xlabel('a')
plt.ylabel('b')
plt.title("first 2 components using iris data")
plt.show()


# In[ ]:

"""
 to find better parameter K, here I use sklearn.grid_search
"""
from sklearn import neighbors as kNN
from sklearn.grid_search import GridSearchCV
from pprint import pprint
import numpy as np

iris = load_iris()    
trainSet, testSet = cross_validation(iris.data, iris.target)

# arbitarily choose 20 random k.
kSet = np.arange(20)+1
knn_parameters = {'n_neighbors': kSet}
clf = sklearn.grid_search.GridSearchCV(kNN.KNeighborsClassifier(), knn_parameters, cv = 10)


clf.fit(zip(*trainSet)[0], zip(*trainSet)[1])

