import pandas as pd
import sys,os
import numpy as np

def get_train_data():

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_train = pd.read_csv(package_directory+'/../../Data/X_train.csv')
    x_train = x_train.drop('id', 1)

    y_train = pd.read_csv(package_directory+'/../../Data/y_train.csv')
    y_train = y_train.drop('id', 1)
    
    x_train = pd.DataFrame(np.ascontiguousarray(x_train))
    y_train = pd.DataFrame(np.ascontiguousarray(y_train), columns=['y'])
                           
    return x_train, y_train

def get_engineered_train_data():

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_train = pd.read_csv(package_directory+'/../../Data/X_Feature_Extracted.csv')

    y_train = pd.read_csv(package_directory+'/../../Data/y_train.csv')
    y_train = y_train.drop('id', 1)
    
    x_train = pd.DataFrame(np.ascontiguousarray(x_train))
    y_train = pd.DataFrame(np.ascontiguousarray(y_train), columns=['y'])
                           
    return x_train, y_train

def get_test_data():
    package_directory = os.path.dirname(os.path.abspath(__file__))

    x_test = pd.read_csv(package_directory+'/../../Data/X_test.csv')
    x_test = x_test.drop('id', 1)
    
    x_test = pd.DataFrame(np.ascontiguousarray(x_test))
                           
    return x_test


