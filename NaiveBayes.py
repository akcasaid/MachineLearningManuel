import numpy as np

class Naive_Bayes:
# alpha:             Just remember this as a smoothing parameter. Will explain it's importance later.
# classes:           To store the unique class labels.
# class_prior_probs: This stores the probabilities of an individual classes
# feature_probs:     This stores the feature probabilities or conditional probability.
    def __init__(self,alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_prior_probs = {}
        self.feature_probs = {}


    def fit(self,X,y):
#Define the fit method to train the Naive Bayes classifier.
#Here "X" is the feature matrix that contains the data samples
#"y" contains the class labels corresponding to each data sample.
#Also get the unique class labels and store them in "self.classes".

        self.classes_np.unique(y)

#Now loop over each class.
#And from "X" subset the features corresponding to the current class "c".
#Using that calculate the probability of that class "c" occurring in the training data.
#This is calculated by:
#(Total No of times class "c" occurred)/Total No of classes

        for c in self.classes:
            X_c = X[y==c]
            self.classes_prior_probs[c]=(len(X_c)+ self.alpha) / (len(X)+self.alpha * len(self.__classes)) 
            #Here you can see we have multiple by "alpha". For example if you have unseen feature data. Since it is unseen, the algorithm calculates it's probability as 0. By multiplying this with the probability the entire value is 0.To  void this alpha is used whose default value is 1.
            self.feature_probs[c]=(np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c)+self.alpha * X.shape[1])
            # Similarly calculate the probabilities of individual features for the current class "c". To simplify this is the probability of occurrence of that feature given that the class is c.


#_predict_single  
#This method predicts the class label for a single instance. This is done by calculating posterior probability
#This is a conditional probability that we get after updating the previous probabilities. Now return the class label with the highest probability.
    def _predict_single(self,x):
        posteriors = {c: np.log(self.class_prior_ptobs[c]) + np.sum(np.log(self.feature_probs[c]) * x) for c in self.classes}
        return max (posteriors, key = posteriors.get)
    


#Finally define the "predict" method to make the predictions on a set of data. For each data in "X" it runs the above "_predict_single" method and returns the class label.
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return predictions

    

