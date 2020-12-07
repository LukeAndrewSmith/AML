import pandas as pd
import sys,os
import numpy as np

def get_train_data():

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_train_eeg_1 = pd.read_csv(package_directory+'/../../Data/train_eeg1.csv')
    x_train_eeg_1 = x_train_eeg_1.drop('Id', axis=1)

    x_train_eeg_2 = pd.read_csv(package_directory+'/../../Data/train_eeg2.csv')
    x_train_eeg_2 = x_train_eeg_2.drop('Id', axis=1)
    
    x_train_emg = pd.read_csv(package_directory+'/../../Data/train_emg.csv')
    x_train_emg = x_train_emg.drop('Id', axis=1)

    y_train = pd.read_csv(package_directory+'/../../Data/train_labels.csv')
    y_train = y_train.drop('Id', axis=1)
      
    return x_train_eeg_1, x_train_eeg_2, x_train_emg, y_train
#     return x_train, y_train

def get_test_data():
    
    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_test_eeg_1 = pd.read_csv(package_directory+'/../../Data/test_eeg1.csv')
    x_test_eeg_1 = x_test_eeg_1.drop('Id', axis=1)

    x_test_eeg_2 = pd.read_csv(package_directory+'/../../Data/test_eeg2.csv')
    x_test_eeg_2 = x_test_eeg_2.drop('Id', axis=1)
    
    x_test_emg = pd.read_csv(package_directory+'/../../Data/test_emg.csv')
    x_test_emg = x_test_emg.drop('Id', axis=1)
                           
    return x_test_eeg_1, x_test_eeg_2, x_test_emg
