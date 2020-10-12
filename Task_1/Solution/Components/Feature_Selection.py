from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLarsCV
import pandas as pd
import numpy as np

class Feature_Selection:
    def select_lasso(X,y):
        lasso = LassoLarsCV(normalize=True, max_iter=1000).fit(X,y)
        selector = SelectFromModel(lasso, prefit=True)
        return selector.transform(X)
        