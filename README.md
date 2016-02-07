Practice collection of machine learning algorithms
========

## ■ Update 2016/02/06
- k nearst neighbors  
 - An implementation of kNN algorithm using sklearn.datasets  
 - Also, a practice to k-largest selection question by using randomized selection algorithm  
  - [reference machine-learning-under-the-hood-writing-your-own-k-nearest-neighbour-algorithm/](http://blog.cambridgecoding.com/2016/01/16/machine-learning-under-the-hood-writing-your-own-k-nearest-neighbour-algorithm/)  
 - [reference k-nearest-neighbors-and-curse-of-dimensionality-in-python-scikit-learn/](http://bigdataexaminer.com/uncategorized/k-nearest-neighbors-and-curse-of-dimensionality-in-python-scikit-learn/)
 - Main steps are described as belows  
> 1. calculate dist. between any two points  
> 2. find the nearest neighbors base on their pairwise distance  
> 3. majority vote on a class labels based on nearest neighbors list  
 
 - amendment 2016-02-06  
>1. some drawback: for convenient reasons, here  
    (a) <font color="red">I didn't normalize the raw_data on dataset iris</font>  
    (b) <font color="red">use holdout CV instead of k-fold</font>  
    (c) <font color="red">I didn't evaluate the effect of dimension to kNN</font>  
>2. ploted projections of first two principle compoenets by SVD (singular value decomposition) and PCA (principle component analysis). Both show few dimensions adresses well. ([see relation between SVD and PCA](http://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca))  
>3. updated source code.  

<p align="center">
  <img src="https://dl.dropboxusercontent.com/u/23983489/pca_iris.png" />
</p>

## ■ Update 2016/02/05
- naive bayes classifier  
-- A basic example of naive bayes classifier for document abuse classification  