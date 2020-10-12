from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd

def select():
    print("Hi")

def gaussian(X):
    print(type(X))
    #c.f. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
    trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
    qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
    return pd.DataFrame(qt.fit_transform(X)) # Transforms X, and python variables are passed by reference so result is seen outside
