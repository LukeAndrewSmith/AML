from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
import pandas as pd
import numpy as np

def mean(X, x_supp=None):
    if x_supp is not None:
        x_supp.columns = X.columns
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def median(X, x_supp=None):
    if x_supp is not None:
        x_supp.columns = X.columns    
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def iterative_regression(X,  x_supp=None, n_nearest_features=10): # default regressor: BayesianRidge
    if x_supp is not None:
        x_supp.columns = X.columns     
    imp = IterativeImputer(missing_values=np.nan, 
                            max_iter=10, initial_strategy='median',random_state=0, n_nearest_features=n_nearest_features)
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def knn(X, x_supp=None, neighbors=1):
    if x_supp is not None:
        x_supp.columns = X.columns     
    imp = KNNImputer(missing_values=np.nan, weights='distance')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X))

def missing_values_mask(X):
    indicator = MissingIndicator(features='all')
    return indicator.fit_transform(X)


def mean2(X, x_supp):
    if x_supp is not None:
        x_supp.columns = X.columns
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X)), pd.DataFrame(imp.transform(x_supp))

def median2(X, x_supp):
    if x_supp is not None:
        x_supp.columns = X.columns    
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X)), pd.DataFrame(imp.transform(x_supp))

def iterative_regression2(X,  x_supp, n_nearest_features=10): # default regressor: BayesianRidge
    if x_supp is not None:
        x_supp.columns = X.columns     
    imp = IterativeImputer(missing_values=np.nan, 
                            max_iter=10, initial_strategy='median',random_state=0, n_nearest_features=n_nearest_features)
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X)), pd.DataFrame(imp.transform(x_supp))

def knn2(X, x_supp, neighbors=1):
    if x_supp is not None:
        x_supp.columns = X.columns     
    imp = KNNImputer(missing_values=np.nan, weights='distance',n_neighbors=neighbors)
    imp.fit(pd.concat([X,x_supp],ignore_index=True))
    return pd.DataFrame(imp.transform(X)), pd.DataFrame(imp.transform(x_supp))

