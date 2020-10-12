from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
import pandas as pd
import numpy as np

def mean(X, x_supp=None):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def median(X, x_supp=None):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def iterative_regression(X,  x_supp=None): # default regressor: BayesianRidge
    imp = IterativeImputer(missing_values=np.nan, 
                            max_iter=10, initial_strategy='median',random_state=0)
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def knn(X, x_supp=None):
    imp = KNNImputer(missing_values=np.nan, weights='distance')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def missing_values_mask(X):
    indicator = MissingIndicator(features='all')
    return indicator.fit_transform(X)




