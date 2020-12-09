import numpy as np
import pandas as pd
from pprint import pprint
from mne_features.feature_extraction import extract_features
from mne_features.univariate import get_univariate_funcs
from mne_features.bivariate import get_bivariate_funcs

def get_features(X=None, types=['univariate', 'bivariate'],verbose=False,precomputed=None):
    X_new = []
    if 'univariate' in types:
        X_new.append(__univariate_features(X=X,precomputed=precomputed))
    if 'bivariate' in types:
        X_new.append(__bivariate_features(X=X,precomputed=precomputed))
    return pd.concat(X_new,axis=1,ignore_index=True)

def __univariate_features(X=None,precomputed=None):
    if precomputed == 'train':
        return pd.read_csv('../../Data/univariate_features_train.csv')
    elif precomputed == 'test':
        return pd.read_csv('../../Data/univariate_features_test.csv')
    else:
        params = {'pow_freq_bands__freq_bands':    np.array([.5, 4, 8, 13, 30]),
                  'energy_freq_bands__freq_bands': np.array([.5, 4, 8, 13, 30])}
        X_new = extract_features(X, 128, get_univariate_funcs(128), funcs_params=params, n_jobs=-1)
        return pd.DataFrame(X_new)

def __bivariate_features(X=None,precomputed=None):
    if precomputed == 'train':
        return pd.read_csv('../../Data/bivariate_features_train.csv')
    elif precomputed == 'test':
        return pd.read_csv('../../Data/bivariate_features_test.csv')
    else:
        X_new = extract_features(X, 128, get_bivariate_funcs(128), n_jobs=-1)
        return pd.DataFrame(X_new)