klearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def mean(x):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
    return pd.DataFrame(fill_NaN.fit_transform(x))

def median(x):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
    return pd.DataFrame(fill_NaN.fit_transform(x))

def impute():
    print("Hi")
