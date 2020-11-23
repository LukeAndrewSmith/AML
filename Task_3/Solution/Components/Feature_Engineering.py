import numpy as np
import pandas as pd
import neurokit2 as nk
from multiprocessing import Pool
from pprint import pprint

def extract_features(X=None):
    X_2 = []
    index_failed = []
    for index, row in X.iterrows():
        if index % 100 == 0:
            print(index)
        try:
            signals,_ = nk.ecg_process(row.dropna(), sampling_rate=300)
            peak_types = ["ECG_P_Peaks","ECG_Q_Peaks","ECG_R_Peaks","ECG_S_Peaks","ECG_T_Peaks"]
            peaks = [list(signals.index[signals[peak_type]==1]) for peak_type in peak_types]
            ecg_clean = signals[['ECG_Clean']]
            X_2.append(extract_features_peaks(index,peaks,ecg_clean))
        except Exception as inst:
            print("Index failed: ",index, "with error: \n", inst)
            index_failed.append(index)
            X_2.append(extract_features_peaks(index,[]))
            continue
        
    return pd.DataFrame(X_2)

def extract_features_parrallel(X=None):
    X_2 = []
    index_failed = []
    threads = list()
    
    with Pool(processes=16) as pool:
        tuples = [(index,row.dropna().tolist()) for index,row in X.iterrows()]
        X_2 = pool.starmap(process_row,tuples)
    X_2 = [x if x is not None or [] else [0]*len(X_2[0]) for x in X_2] # Put zeros where ecg_process failed
    return pd.DataFrame(X_2)
        
def process_row(index,row):
    try:
        signals,_ = nk.ecg_process(row, sampling_rate=300)
        peak_types = ["ECG_P_Peaks","ECG_Q_Peaks","ECG_R_Peaks","ECG_S_Peaks","ECG_T_Peaks"]
        peaks = [list(signals.index[signals[peak_type]==1]) for peak_type in peak_types] 
        ecg_clean = signals[['ECG_Clean']]
    except Exception as inst:
        # NeuroKit2 Problems: sometimes it fails miserably during nk.ecg_process() for weird reasons that by my
        # investigation shouldn't be occuring.
        # TODO: FIND A BETTER LIBRARY (for now just replace with zeros)
        print(index, " failed with error: ", inst)
        peaks = None
        ecg_clean = None
    return extract_features_peaks(index,peaks,ecg_clean)
    

def extract_features_peaks(index,peaks,ecg_clean):

    if peaks is None or ecg_clean is None:
        return None
    
    features = []
    
    ################
    # Random Metrics
    ################
    if np.mean(np.diff(peaks[2])) != 0:
        bpm = 18000 / np.mean(np.diff(peaks[2]))
    else:
        bpm = 0
    features.append(bpm)
    
    ##########################
    # Single-peak type metrics
    ##########################
    peak_heights = [[x[0] for x in ecg_clean.iloc[p].to_numpy()] for p in peaks]

    # Location
    for i in range(len(peaks)):
        peak = peaks[i]
        diff = np.diff(peak)
        features.append(np.mean(diff))
        features.append(np.var(diff))
        
    # Scale
    for i in range(len(peak_heights)):
        features.append(np.mean(np.abs(peak_heights[i]))) # Mean absolute
        features.append(np.array([1 for x in np.sign(peak_heights[i]) if x == -1]).sum()) # #postive values
        features.append(np.array([1 for x in np.sign(peak_heights[i]) if x == 1]).sum()) # #negative values
        features.append(np.var(peak_heights[i]))
        
    # TODO:
    # 1. Mean absolute value
    # 2. Number of positive and negative values
    # 3. Mean doesn't make sense
    # 4. Base the scale metrics below on similar thinking (mean absolute difference etc.)
        
    ##########################
    # Multi-peak type metrics
    ##########################
        # For the multi-peak type measure we require the combination of each peak identified at index i
        # to represent a valid heartbeat <p,q,r,s,t> so we clean the data
    peaks = peaks_cleaning(peaks)
    peak_heights = [[x[0] for x in ecg_clean.iloc[p].to_numpy()] for p in peaks]

    # Location
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            if i != j:
                diff = np.array(peaks[i])-np.array(peaks[j])
                features.append(np.mean(diff))
                features.append(np.var(diff))
            
    # Scale
    for i in range(len(peak_heights)):
        for j in range(len(peak_heights)):
            if i != j:
                diff = np.array(peak_heights[i])-np.array(peak_heights[j])
                features.append(np.mean(diff))
                features.append(np.var(diff))
    
    ##########
    # Problems
    ##########
    
    # NeuroKit2 Problems: sometimes it only identifies a single peak for one of the p,q,r,s,t peaks, in which 
    # case a diff will result in [] and so NaNs when mean/var is calculated
    # TODO: FIND A BETTER LIBRARY (for now just replace with zeros)
    if np.isnan(np.min(np.array(features))):
        features = [x if not np.isnan(x) else 0 for x in features]
        print(index, ": NaN in features")
    
    return features


def peaks_cleaning(peaks):
    # This method identifies tuples <p,q,r,s,t> which represent a single valid heartbeat,
        # It therefore removes cases such as <_,_,r,s,t>, where p and q are missing
        # It is robust to potential errors in the middle of the peak arrays
    # Return:
        # It return an array, A, of equal length arrays of peak location for p,q,r,s,t peaks
        # A[:][i] represents a valid heartbeat tuple <p,q,r,s,t>

    # median r_peak distance used later to check if the p_peak r_peak distance in valid
    med = np.median(np.diff(peaks[0])) # TODO: MAKE SURE THIS IS VALID
    
    # Obtain valid <p,q,r,t> pairs
    # assume p,q,r,t each ordered asc.
        # for each possible <p,q,r,t>
            # if p>q,r,t then we can never for a valid pair with q,r,t so drop q,r,t
                # repeat until p<q,r,t
                # same with p<q>r,t, p<q<r>t,
            # cont. until p<q<r<t, then check that p<q<r<t is feasible
    i = -1
    while i < len(peaks[0]):
        i += 1
        if i <= np.min(list(map(len, peaks)))-1:
            for j in range(len(peaks)-2):
                while (peaks[j][i] > peaks[j+1][i]):
                    peaks[j+1].pop(i)
                    if i >= len(peaks[j+1]): # Popping so check index i valid
                        break
            if i <= np.min(list(map(len, peaks)))-1: # Popping so check index i valid
                if peaks[-1][i] - peaks[0][i] > med:
                    peaks[0].pop(i)
                    i -= 1
        else:
            for i in range(len(peaks)):
                peaks[i] = peaks[i][:np.min(list(map(len, peaks)))]
            break
    return peaks