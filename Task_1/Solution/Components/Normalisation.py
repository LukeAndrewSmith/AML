from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd

def select():
    print("Hi")

def gaussian(X):
    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    return pd.DataFrame(qt.fit_transform(X),columns=X.columns)

def yeo_johnson(X, standardize=True):
    yeo = PowerTransformer(standardize=standardize)
    return pd.DataFrame(yeo.fit_transform(X), columns=X.columns)
