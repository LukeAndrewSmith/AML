import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from collections import Counter
from random import randint


class Smoothing_Classifier(BaseEstimator):
    """
    This classifier does some "smoothing" on the predicted classes of another classifier (assuming temporal coherence)
    It takes the prediction of each epoch and some surrounding predictions as features
    """

    def __init__(self, clf, lag = 5, border_vote = 5, verbose=False):
        self.clf = clf
        self.lag = lag
        self.border_vote=5
        self.verbose = verbose

    def fit(self, y_ind_pred, y):
 
        X = self.build_features(y_ind_pred)        
        # feed the new features into a classifier
        self.clf.fit(X,y)
        
        return self


    def predict(self, y_ind_pred):
        X = self.build_features(y_ind_pred)
        
        y_smoothed = self.clf.predict(X)
        
        return y_smoothed

    
    def build_features(self, y_ind_pred):
        # build a new design matrix X of the given individual predictions for each epoch 
        #border handling: the lag missing values before and after the scope of the time series are supposed to be that of the majority of the first/ last border_vote predictions
        y_ind_pred = y_ind_pred.reshape(-1)
        X = pd.DataFrame(y_ind_pred)
        # earlier values
        start_border = np.bincount(y_ind_pred[:self.border_vote-1]).argmax()
        for f in range(-1* self.lag, 0):
            X[f] = X[0].shift(periods=-f, fill_value = start_border)
        end_border = np.bincount(y_ind_pred[-1*self.border_vote:]).argmax()
        # later values
        for f in range(1,self.lag + 1):
            X[f] = X[0].shift(periods=-f, fill_value = end_border)
        return X