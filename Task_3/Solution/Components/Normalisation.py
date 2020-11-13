from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import numpy as np
import pandas as pd

def select():
    print("Hi")

def gaussian(X,n_quants=50):
    qt = QuantileTransformer(n_quantiles=n_quants, output_distribution='normal')
    return pd.DataFrame(qt.fit_transform(X),columns=X.columns)

def standard_scaler(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
