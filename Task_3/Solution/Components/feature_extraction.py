import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_extraction as tsfe


def get_features(X=None, types=['heartbeats'],verbose=False,precomputed=None):
    X_new = []
    if 'heartbeats' in types:
        X_new.append(__heartbeats(X,verbose=verbose,precomputed=precomputed))
    if 'timeseries' in types:
        X_new.append(__timeseries(X,verbose=verbose,precomputed=precomputed))
    return pd.concat(X_new,axis=1,ignore_index=True)

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
        X_new = pd.read_csv('../../Data/ts_feat_train.csv').drop('id', 1)
    elif precomputed == 'test':
        X_new = pd.read_csv('../../Data/ts_feat_test.csv').drop('id', 1)
    else:
        X_long = X.reset_index().melt(id_vars='index',var_name = "time")
        # These features and their corresponding parameters seem to be promising
        kind_to_fc_parameters = {'value': {'agg_linear_trend': [{'attr': 'stderr','chunk_len': 10,'f_agg': 'max'},
                                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
                                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'}],
                                        'autocorrelation': [{'lag': 9}],
                                        'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.2},
                                                            {'f_agg': 'mean', 'isabs': True, 'qh': 0.8, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.4, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': False, 'qh': 0.8, 'ql': 0.6},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.2},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.4},
                                                            {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.4}],
                                        'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                                        'quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.7}, {'q': 0.8}, {'q': 0.9}],
                                        'range_count': [{'max': 1, 'min': -1}],
                                        'ratio_value_number_to_time_series_length': None,
                                        'value_count': [{'value': 0}],
                                        'variance': None}}
        X_new = tsfe.extract_features(timeseries_container = X_long.dropna(),kind_to_fc_parameters=kind_to_fc_parameters,
                                         column_id="index",column_sort="time", n_jobs= 16)
        impute(X_new)
    return X_new

    


