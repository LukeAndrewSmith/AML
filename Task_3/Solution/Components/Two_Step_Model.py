import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from collections import Counter
from random import randint


class TwoStepModel(BaseEstimator):
    """
    Class for a Classifier that classifies data in two steps: First step classifies all classes vs one noisy class and the 2nd
    one does the classification between the leftover classes. A dict for both models with all arguments need to be passed to
    enable flexible model params (general use remarks: no brackets for clftype e.g. SVC, not SVC() as argument)
    """

    def __init__(self, clf1=SVC(), clf2=SVC(), verbose=False):
        # other
        self.clf1 = clf1
        self.clf2 = clf2
        self.verbose = verbose

    def fit(self, X, y):

        # prepare the first dataset in changin y==1, y==2 to y= 0:
        X_1 = X.copy()  # no changes on X, just to be consistent
        y_1 = y.copy()
        y_1['y'] = np.where((y_1.y == 1), 0, y_1.y)
        y_1['y'] = np.where((y_1.y == 2), 0, y_1.y)

        # train the first model

        if self.verbose:
            print('Training first task with clf1: ')
            print(self.clf1)

        self.clf1 = self.clf1.fit(X_1, y_1)

        # this time remove all datapoints where y==3
        X_2 = X.copy()  # no changes on X, just to be consistent
        y_2 = y.copy()
        X_2 = X_2[y['y'] != 3]
        y_2 = y_2[y['y'] != 3]

        if self.verbose:
            print('Training second task with clf2: ')
            print(self.clf2)

        self.clf2 = self.clf2.fit(X_2, y_2)
        return self

    def most_common_classes(self, lst):
        # Given: array of arrays of predictions for each estimators
        lst = np.transpose(np.array(lst))  # we want array of arrays of predictions for each instance in X
        data = [Counter(sub_lst) for sub_lst in lst]
        # Get the most common class predictions (could be multiple)
        mc_dup = [[(x[0], x[1]) for x in sub_data.most_common() if sub_data.most_common()[0][1] == x[1]] for sub_data in
                  data]
        # If multiple most common class predictions, choose random (if len(x)=1 then we chose randint(0,0)=0)
        mc = [x[randint(0, len(x) - 1)][0] for x in mc_dup]
        return mc

    def predict(self, X):
        # let both models predict and classify as 3 if clf1 says so otherwise take clf2 prediction
        y_pred = list()
        y_pred1 = self.clf1.predict(X)
        y_pred2 = self.clf2.predict(X)

        for i, y in enumerate(y_pred1):
            if y == 3:
                y_pred.append(y)
            else:
                y_pred.append(y_pred2[i])
        return y_pred
