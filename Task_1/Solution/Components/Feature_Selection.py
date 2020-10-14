from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLarsCV
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

def select_lasso(X,y):
    lasso = LassoLarsCV(normalize=True, max_iter=1000).fit(X,y)
    selector = SelectFromModel(lasso, prefit=True)
    return selector.transform(X)
    
def select_percentile_regr(X,y,percent=50):
    X = SelectPercentile(f_regression, percent).fit_transform(X, y)
    return X # TODO: also return which columns were deleted so that we can remove them from X_test

def select_percentile_mut_inf(X,y,percent=50):
    X = SelectPercentile(mutual_info_regression, percent).fit_transform(X, y)
    return X # TODO: also return which columns were deleted so that we can remove them from X_test