import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from collections import Counter
from random import randint


class Classifying_Transformer(BaseEstimator):
    """
    Wrapper class for a classifier acting as a transformer (in order to use it as an intermediate step in a pipeline)
    """

    def __init__(self, clf, verbose=False):
        self.clf = clf
        self.verbose = verbose

    def fit(self, X, y):

        self.clf.fit(X,y)
        
        return self


    def transform(self, X):
        # let both models predict and classify as 3 if clf1 says so otherwise take clf2 prediction
        y_pred = self.clf.predict(X).reshape(-1, 1)
        
        return y_pred
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
