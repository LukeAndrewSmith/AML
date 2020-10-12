import pandas as pd
import numpy as np

def knn(x):
    # x=...
    # return x
    print("Hi")

def _get_cov(X):
    # calculates cov matrix
    from sklearn.covariance import EmpiricalCovariance
    cov = EmpiricalCovariance().fit(X)
    return cov

def mahalanobis_distance(X_train, y_train, alpha=3, verbose=False):
    #join x train and y train
    train_all = X_train.copy()
    train_all['age'] = y_train
    numpy_all = train_all.to_numpy()
    
    # calc cov matric
    cov = _get_cov(numpy_all)
    mahal = cov.mahalanobis(train_all)
    mahal_mean = np.mean(mahal)
    mahal_std = np.std(mahal)

    # get all id that have higher z score than alpha and remove them
    outliers = np.where((np.asarray(mahal) - mahal_mean) / mahal_std > alpha)
    if verbose:
        print('# Outliers: ', outliers)
    
    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])

    return X,y


def dbscan(X_train, y_train, verbose=False):
    from sklearn.cluster import DBSCAN
    train_all = X_train.copy()
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    labels = clustering.labels_
    if verbose:
        print(labels)

    outliers = np.where((np.asarray(labels) == -1))
    if verbose:
        print('# Outliers: ', outliers)

    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])

    return X,y