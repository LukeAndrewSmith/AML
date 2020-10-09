import pandas as pd
import sys,os

def get_data():

    package_directory = os.path.dirname(os.path.abspath(__file__))
    print(package_directory)

    x_train = pd.read_csv(package_directory+'/../../Data/X_train.csv')
    x_train = x_train.drop('id', 1)

    y_train = pd.read_csv(package_directory+'/../../Data/y_train.csv')
    y_train = y_train.drop('id', 1)

    x_test = pd.read_csv(package_directory+'/../../Data/X_test.csv')
    x_test = x_test.drop('id', 1)
    
    return x_train, y_train, x_test