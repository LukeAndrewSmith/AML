import numpy as np
import pandas as pd
from pprint import pprint
from mne_features.feature_extraction import extract_features
from mne_features.univariate import get_univariate_funcs
from mne_features.multivariate import get_multivariate_funcs

def extract_features(X=None, types=['univariate', 'multivariate'],verbose=False,precomputed=None):
    X_new = []
    if 'univariate' in types:
        X_new.append(__univariate_features(X=X,precomputed=precomputed))
    if 'multivariate' in types:
        X_new.append(__multivariate_features(X=X,precomputed=precomputed))
    return pd.concat(X_new,axis=1,ignore_index=True)

def __univariate_features(X=None,precomputed=None):
    if precomputed = 'train':
        return pd.read_csv('../../Data/univariate_features_train.csv')
    else if precomputed = 'test':
        return pd.read_csv('../../Data/univariate_features_test.csv')
    else:
        params = {'pow_freq_bands__freq_bands':    np.array([.5, 4, 8, 13, 30]),
                  'energy_freq_bands__freq_bands': np.array([.5, 4, 8, 13, 30])}
        return extract_features(X, 128, get_univariate_funcs(128), funcs_params=params, n_jobs=-1)

def __multivariate_features(X=None,precomputed=None):
    if precomputed = 'train':
        return pd.read_csv('../../Data/multivariate_features_train.csv')
    else if precomputed = 'test':
        return pd.read_csv('../../Data/multivariate_features_test.csv')
    else:
        return extract_features(X, 128, get_multivariate_funcs(128), funcs_params=params, n_jobs=-1))