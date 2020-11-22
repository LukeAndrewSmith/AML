import numpy as np
import pandas as pd
import neurokit2 as nk
from multiprocessing import Pool

# def extract_features(X=None):
#     X_2 = []
#     index_failed = []
#     for index, row in X.iterrows():
#         if index % 100 == 0:
#             print(index)
#         try:
#             signals,_ = nk.ecg_process(row.dropna(), sampling_rate=300)
#         except:
#             print("Index failed: ",index)
#             index_failed.append(index)
#             continue
#         peak_types = ["ECG_P_Peaks","ECG_Q_Peaks","ECG_R_Peaks","ECG_S_Peaks","ECG_T_Peaks"]
#         peaks = [list(signals.index[signals[peak_type]==1]) for peak_type in peak_types]
#         X_2.append(extract_features_peaks(peaks))
#     return pd.DataFrame(X_2)

def extract_features_parrallel(X=None):
    X_2 = []
    index_failed = []
    threads = list()
    
    with Pool(processes=16) as pool:
        tuples = [(index,row.dropna().tolist()) for index,row in X.iterrows()]
        X_2 = pool.starmap(process_row,tuples)
    print(type(X_2))
    print(type(X_2[0]))
#     print(X_2)
    return X_2
#     return pd.DataFrame(X_2)
        
def process_row(index,row):
    try:
        signals,_ = nk.ecg_process(row.dropna(), sampling_rate=300)
        peak_types = ["ECG_P_Peaks","ECG_Q_Peaks","ECG_R_Peaks","ECG_S_Peaks","ECG_T_Peaks"]
        peaks = [list(signals.index[signals[peak_type]==1]) for peak_type in peak_types] 
#         if index % 100 == 0:
#             print("Done: ", index)
        return [extract_features_peaks(peaks)]
    except:
        print("Index failed: ",index)
#         index_failed.append(index)


def extract_features_peaks(peaks):

    features = []
    
    bpm = 18000 / np.mean(np.diff(peaks[2]))
    features.append(bpm)
    
    ######################
    # Single-peak type measures
        # For the single peak type measure we wan't all the data that ecg_process found
        
    # Variablity measures: single peak type
    for i in range(len(peaks)):
        peak = peaks[i]
        diff = np.diff(peak)
        features.append(np.mean(peak))
        features.append(np.var(peak))
        features.append(np.mean(diff))
        features.append(np.var(diff))
        
    #####################
    # Multi-peak type measures
        # For the multi-peak type measure we require the combination of each peak identified at index i
        # to represent a valid heartbeat <p,q,r,s,t>
    peaks = peaks_cleaning(peaks)
    
    # Variablity measures: multi peak types
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            diff = np.array(peaks[i])-np.array(peaks[j])
            features.append(np.mean(diff))
            features.append(np.var(diff))
            
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
            # cont until p<q<r<t, then check that p<q<r<t is feasible
            
    i = -1
    while i < len(peaks[0]):
        i += 1
        if i <= np.min(list(map(len, peaks)))-1:
            for j in range(len(peaks)-1):
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