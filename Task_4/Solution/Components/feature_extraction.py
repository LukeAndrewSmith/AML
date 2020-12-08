import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_extraction as tsfe
import neurokit2 as nk
from multiprocessing import Pool
from pprint import pprint

def get_features(X=None, types=[],verbose=False,precomputed=None):
    X_new = []
    if 'heartbeats' in types:
        #X_new.append(__heartbeats(X,verbose=verbose,precomputed=precomputed))
        pass
    return pd.concat(X_new,axis=1,ignore_index=True)