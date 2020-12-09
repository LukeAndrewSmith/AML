import pandas as pd
import sys,os
import numpy as np

def get_train_data_separate_signals(per_subject=False)):

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_train_eeg_1 = pd.read_csv(package_directory+'/../../Data/train_eeg1.csv')
    x_train_eeg_1 = x_train_eeg_1.drop('Id', axis=1)

    x_train_eeg_2 = pd.read_csv(package_directory+'/../../Data/train_eeg2.csv')
    x_train_eeg_2 = x_train_eeg_2.drop('Id', axis=1)

    x_train_emg = pd.read_csv(package_directory+'/../../Data/train_emg.csv')
    x_train_emg = x_train_emg.drop('Id', axis=1)

    y_train = pd.read_csv(package_directory+'/../../Data/train_labels.csv')
    y_train = y_train.drop('Id', axis=1)

    if per_subject:
        return split_df_per_subject(x_train_eeg_1), split_df_per_subject(x_train_eeg_2), 
                split_df_per_subject(x_train_emg), split_df_per_subject(y_train)
        
    return x_train_eeg_1, x_train_eeg_2, x_train_emg, y_train

def get_train_data():
    x_train_eeg_1, x_train_eeg_2, x_train_emg, y_train = get_train_data_separate_signals()
    x_train = np.stack((x_train_eeg_1,x_train_eeg_2,x_train_emg),axis=1)
    return x_train, y_train

def get_test_data_separate_signals():

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_test_eeg_1 = pd.read_csv(package_directory+'/../../Data/test_eeg1.csv')
    x_test_eeg_1 = x_test_eeg_1.drop('Id', axis=1)

    x_test_eeg_2 = pd.read_csv(package_directory+'/../../Data/test_eeg2.csv')
    x_test_eeg_2 = x_test_eeg_2.drop('Id', axis=1)

    x_test_emg = pd.read_csv(package_directory+'/../../Data/test_emg.csv')
    x_test_emg = x_test_emg.drop('Id', axis=1)

    return x_test_eeg_1, x_test_eeg_2, x_test_emg

def get_test_data():
    x_test_eeg_1, x_test_eeg_2, x_test_emg = get_test_data_separate_signals()
    x_train = np.stack((x_test_eeg_1,x_test_eeg_2,x_test_emg),axis=1)
    return x_train

def split_df_per_subject(df):
    #returns a list of three dfs split by subject 
    df_1 = df.iloc(:21599)
    df_2 = df.iloc(21600:43199)
    df_3 = df.iloc(43200:)
    return [df_1, df_2, df_3]