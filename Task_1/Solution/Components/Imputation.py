from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def mean(x):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
    return pd.DataFrame(fill_NaN.fit_transform(x))

def median(x):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
    return pd.DataFrame(fill_NaN.fit_transform(x))

def impute():
    print("Hi")
