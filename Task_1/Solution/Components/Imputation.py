from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
import pandas as pd
import numpy as np

def mean(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    return pd.DataFrame(imp.fit_transform(X))

def median(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    return pd.DataFrame(imp.fit_transform(X))

def iterative_regression(X):
    imp = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, 
                            max_iter=10, initial_strategy='median',random_state=0)
    return pd.DataFrame(imp.fit_transform(X))

def knn(X):
    imp = KNNImputer(missing_values=np.nan, weights='distance')
    return pd.DataFrame(imp.fit_transform(X))

def original_values_mask(X):
    


