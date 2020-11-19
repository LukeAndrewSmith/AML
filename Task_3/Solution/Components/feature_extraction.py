import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg

def get_features(X=None, types=['heartbeats'],verbose=False):
    X_new = []
    if 'heartbeats' in types:
        X_new.append(__heartbeats(X,verbose=verbose))
    return pd.concat(X_new,axis=1,ignore_index=True)

def __heartbeats(X=None,verbose=False):
    X_new = []
    for index, row in X.iterrows(): 
        _, _, peaks, _, templates, _, _ = ecg.ecg(signal=row.dropna(), sampling_rate=300.0, show=False)
        #get one averaged heartbeat template for each time series
        average = np.mean(templates, axis=0)
        #calculate the variances of the heartbeat templates for a selected number of points (evenly distributed)
        sel_templates = templates[np.round(np.linspace(0, len(templates)-1, 20)).astype(int)]
        variances = np.var(sel_templates,axis=0)
        #calculate the distances between r-peaks
        peaks_distances = np.diff(peaks)
        mean_peaks_distances = np.mean(peaks_distances)
        var_peaks_distances = np.var(peaks_distances)
        features = np.concatenate([average,variances,[mean_peaks_distances,var_peaks_distances]])
        X_new.append(features)
        if verbose and index % 100 == 0:
            print(index)
            print(X_new)
    #X_new = pd.concat(X_new, ignore_index=True)
    X_new = pd.DataFrame(X_new)
    return X_new

    


