import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from collections import Counter
from random import randint


class Neighbour_Features(BaseEstimator):
    """
    This transformer takes a given feature matrix and concatenates the features of temporally neighbored epochs
    """

    def __init__(self, lag = 5, verbose=False):
        self.lag = lag
        self.verbose = verbose

    def fit(self, X, y):        
        return self


    def transform(self, X):
        # build a new design matrix X of the given one by adding to each row the features of the lag rows before and after of that specific row
        #border handling: the lag missing rows before and after the scope of the time series are supposed to be the same as the first/last row
        if self.verbose:
            print(f"shape of X: {X.shape}")
        X = pd.DataFrame(X)
        X_new = []
        n_epochs = 21600
        # subject-wise time series
        for subject_start in range(0,len(X)-n_epochs+1,n_epochs):
            X_subject = X.iloc[subject_start:(subject_start + n_epochs)]
            if self.verbose:
                print(f"####################################subject {subject_start/n_epochs}###########################################")
                print(f"shape of X_subject {subject_start/n_epochs}: {X_subject.shape}")
            feature_set = [X_subject]
            # earlier values            
            for f in range(-1* self.lag, 0):
                shifted = X_subject.shift(periods=-f)
                # replace the first -f rows by the first row with values
                for row in range(-f):
                    shifted.iloc[row]=X_subject.iloc[0]
                if self.verbose:
                    print(f"shape of shifted with lag {f}: {shifted.shape}")
                    print(f"first 5 rows of shifted with lag {f}:")
                    print(shifted.iloc[:5])
                feature_set.append(shifted)
            # later values
            for f in range(1,self.lag + 1):
                shifted = X_subject.shift(periods=-f)
                # replace the last f rows by the last row with values
                for row in range(-f,0):
                    shifted.iloc[row]=X_subject.iloc[-1]
                if self.verbose:
                    print(f"shape of shifted with lag {f}: {X_subject.shape}")
                    print(f"last 5 rows of shifted with lag {f}:")
                    print(shifted.iloc[-5:])
                feature_set.append(shifted)
            subject_features = pd.concat(feature_set,axis=1,ignore_index=True)
            if self.verbose:
                print(f"shape of subject_features of subject {subject_start/n_epochs}: {subject_features.shape}")
            X_new.append(subject_features)
        X_new = pd.concat(X_new, axis=0)
        if self.verbose:
            print(f"shape of X_new: {X_new.shape}")
        return X_new
    
    def fit_transform(self,X,y):
        return self.transform(X)