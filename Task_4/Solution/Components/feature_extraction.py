import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_extraction as tsfe
import neurokit2 as nk
from multiprocessing import Pool
from pprint import pprint

def get_features(X=None, types=['heartbeats'],verbose=False,precomputed=None):
    X_new = []
    if 'heartbeats' in types:
        X_new.append(__heartbeats(X,verbose=verbose,precomputed=precomputed))
    if 'timeseries' in types:
        X_new.append(__timeseries(X,verbose=verbose,precomputed=precomputed))
    if 'peaks' in types:
        X_new.append(__peaks(X,verbose=verbose,precomputed=precomputed))
    if 'hrv' in types:
        X_new.append(__hrv(X,verbose=verbose,precomputed=precomputed))
    return pd.concat(X_new,axis=1,ignore_index=True)

def __hrv(X=None,verbose=False,precomputed=None):
    if precomputed == 'train':
        X_new = pd.read_csv('../../Data/hrv_feat_train.csv').drop('id', 1)
    elif precomputed == 'test':
        X_new = pd.read_csv('../../Data/hrv_feat_test.csv').drop('id', 1)
    else:
        X_new = []
        for index, row in X.iterrows(): 
            _, _, peaks, _, templates, _, _ = ecg.ecg(signal=row.dropna(), sampling_rate=300.0, show=False)
            # heart rate variability features
            hrv = np.array(td.time_domain(rpeaks=peaks,show=False, sampling_rate=300.0, plot=False))
            #features = np.concatenate([average,variances,[mean_peaks_distances,var_peaks_distances]])
            X_new.append(hrv)
        X_new = pd.DataFrame(X_new)
        X_new = X_new[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 18, 19, 24]]
    return X_new

def __heartbeats(X=None,verbose=False,precomputed=None):
    if precomputed == 'train':
        X_new = pd.read_csv('../../Data/heartbeat_feat_train.csv').drop('id', 1)
    elif precomputed == 'test':
        X_new = pd.read_csv('../../Data/heartbeat_feat_test.csv').drop('id', 1)
    else:
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

def __timeseries(X=None,verbose=False,precomputed=None):
    if precomputed == 'train':
        X_new = pd.read_csv('../../Data/ts_feat2_train.csv').drop('id', 1)
    elif precomputed == 'test':
        X_new = pd.read_csv('../../Data/ts_feat2_test.csv').drop('id', 1)
    else:
        X_long = X.reset_index().melt(id_vars='index',var_name = "time")
        # These features and their corresponding parameters seem to be promising
        kind_to_fc_parameters= {'value': {'agg_autocorrelation': [{'f_agg': 'median', 'maxlag': 40}],
                                        'agg_linear_trend': [{'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'},
                                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'max'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'var'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'}],
                                        'autocorrelation': [{'lag': 8}, {'lag': 9}],
                                        'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.0},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.0},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.4},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.6},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.0},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.4, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.6},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.4, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.6}],
                                        'fft_coefficient': [{'attr': 'abs', 'coeff': 4},
                                                            {'attr': 'abs', 'coeff': 6}],
                                        'fourier_entropy': [{'bins': 5}],
                                        'mean_abs_change': None,
                                        'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                                        'quantile': [{'q': 0.1},
                                                    {'q': 0.2},
                                                    {'q': 0.3},
                                                    {'q': 0.6},
                                                    {'q': 0.7},
                                                    {'q': 0.8},
                                                    {'q': 0.9}],
                                        'ratio_value_number_to_time_series_length': None,
                                        'spkt_welch_density': [{'coeff': 2}],
                                        'standard_deviation': None,
                                        'value_count': [{'value': -1}],
                                        'variance': None}}
        # older version:
        # kind_to_fc_parameters = {'value': {'agg_linear_trend': [{'attr': 'stderr','chunk_len': 10,'f_agg': 'max'},
        #                                                     {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'},
        #                                                     {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'},
        #                                                     {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
        #                                                     {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
        #                                                     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
        #                                                     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
        #                                                     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'}],
        #                                 'autocorrelation': [{'lag': 9}],
        #                                 'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.2},
        #                                                     {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
        #                                                     {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
        #                                                     {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
        #                                                     {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.4},
        #                                                     {'f_agg': 'var', 'isabs': False, 'qh': 0.4, 'ql': 0.2},
        #                                                     {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.2},
        #                                                     {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.4},
        #                                                     {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.6},
        #                                                     {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
        #                                                     {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
        #                                                     {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.4}],
        #                                 'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
        #                                 'quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.7}, {'q': 0.8}, {'q': 0.9}],
        #                                 'range_count': [{'max': 1, 'min': -1}],
        #                                 'ratio_value_number_to_time_series_length': None,
        #                                 'value_count': [{'value': 0}],
        #                                 'variance': None}}
        X_new = tsfe.extract_features(timeseries_container = X_long.dropna(),kind_to_fc_parameters=kind_to_fc_parameters,
                                         column_id="index",column_sort="time", n_jobs= 16)
        impute(X_new)
    return X_new

def __peaks(X=None,verbose=False,precomputed=None):
    if precomputed == 'train':
        X_new = pd.read_csv('../../Data/peak_feat_train.csv')
    elif precomputed == 'test':
        X_new = pd.read_csv('../../Data/peak_feat_test.csv')
    else:  
        X_new = []
        with Pool(processes=16) as pool:
            tuples = [(index,row.dropna().tolist(),verbose) for index,row in X.iterrows()]
            X_new = pool.starmap(process_row,tuples)
        X_new = [x if x is not None or [] else [0]*len(X_new[0]) for x in X_new] # Zeros where ecg_process failed
        X_new = pd.DataFrame(X_new)
    return X_new
        
def process_row(index,row,verbose=False):
    try:
        signals,_ = nk.ecg_process(row, sampling_rate=300)
        peak_types = ["ECG_P_Peaks","ECG_Q_Peaks","ECG_R_Peaks","ECG_S_Peaks","ECG_T_Peaks"]
        peaks = [list(signals.index[signals[peak_type]==1]) for peak_type in peak_types] 
        ecg_clean = signals[['ECG_Clean']]
    except Exception as inst:
        # NeuroKit2 Problems: sometimes it fails miserably during nk.ecg_process() for weird reasons that by my
        # investigation shouldn't be occuring.
        # TODO: FIND A BETTER LIBRARY (for now just replace with zeros)
        if verbose:
            print(index, " failed with error: ", inst)
        peaks = None
        ecg_clean = None
    return extract_features_peaks(index,peaks,ecg_clean,verbose)
    
def extract_features_peaks(index,peaks,ecg_clean,verbose=False):

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
        if verbose:
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