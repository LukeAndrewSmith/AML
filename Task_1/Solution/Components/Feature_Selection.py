from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLarsCV, Lasso
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE

def select_lasso(X,y):
    lasso = LassoLarsCV(normalize=True, max_iter=1000).fit(X,y)
    selector = SelectFromModel(lasso, prefit=True)
    return selector.transform(X)
    
def select_percentile_regr(X,y,percent=50):
    X = SelectPercentile(f_regression, percent).fit_transform(X, y)
    return X # TODO: also return which columns were deleted so that we can remove them from X_test

def select_percentile_mut_inf(X,y,x_test=None,percent=50):
    fs = SelectPercentile(mutual_info_regression, percent)
    X = fs.fit_transform(X, y)
    if x_test is not None:
        x_test = fs.transform(x_test)
        return X,x_test
    return X # TODO: also return which columns were deleted so that we can remove them from X_test

def drop_correlated(X,x_test=None,verbose=False, percent=0.95):
    # TODO: Pearson and spearman correlation for non linear correlation
    
    # Create correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than percent
    to_drop = [column for column in upper.columns if any(upper[column] > percent)]

    if verbose:
        print(len(to_drop))

    X = X.drop(X[to_drop], axis=1)

    # Drop features 
    if x_test is not None:
        x_test = x_test.drop(x_test[to_drop], axis=1)
        return X, x_test

    return X

def rfe(X,y,x_test):
    selector = RFE(Lasso(), n_features_to_select=30, step=1)
    selector = selector.fit(X, y)
    X_cols = X.columns 
    x_test_cols = x_test.columns 
    selector.transform(X)
    selector.transform(x_test)
    return pd.DataFrame(X, columns=X_cols), pd.DataFrame(x_test, columns=x_test_cols)

# def feature_union(X,x_test=None,verbose=False):
#     return