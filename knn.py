import numpy as np

class knn_classifier:
    def __init__(self, k= 5):
        self.k =k

    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y 

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1- x2) **2))
    
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    
    def _predict_single(self, x):
        #Calculate distance between x and all example in the training set
        distance = [self.euclidean_distance(x,x_train) for x_train in self.X_train]

        #Get the indices of the k-nearest neigbors
        k_induces = np.argsort(distance)[:self.k]

        #Get the labels of the k-nearest neigbors
        k_nearest_labels = [ self.y_train[i] for i in k_induces]

        # Return the most common class label among the k-nearst neigbors 
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common
    
print ("by Said Akça :) ")

