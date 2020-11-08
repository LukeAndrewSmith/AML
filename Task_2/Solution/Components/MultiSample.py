import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from collections import Counter
from random import randint

class MultiDownSyndrome(BaseEstimator):
    
    def __init__(self, clf_type='svc',n_clf=10, max_depth=3,learning_rate=0.1,n_estimators=100 ,verbose=False):
        self.clf_type = clf_type
        self.n_clf = n_clf
        # xgb params
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        # other
        self.verbose = verbose

    def fit(self,X,y):
        if self.verbose:
            print("Training ", self.n_clf, "estimators")
        self.estimators_ = []
        for i in range(self.n_clf):
            rus = RandomUnderSampler(random_state=i)
            X_res, y_res = rus.fit_resample(X, y)
            if self.clf_type=='xgb':
                est = xgb.XGBClassifier(max_depth=self.max_depth,learning_rate=self.learning_rate, \
                                        n_estimators=self.n_estimators).fit(X_res,y_res)
#             elif self.clf_type=='elm':
#                 est = TODO
            elif self.clf_type=='equal':
                if (i<(self.n_clf/2)):
                    est = xgb.XGBClassifier(max_depth=self.max_depth,learning_rate=self.learning_rate, \
                                            n_estimators=self.n_estimators).fit(X_res,y_res)
                else:
                    est = SVC(C=4.25,kernel='rbf',gamma=0.0001,shrinking=True, \
                              cache_size=1000,class_weight='balanced').fit(X_res,y_res)
            elif self.clf_type=='svc-2':
                est = SVC(kernel='rbf',shrinking=True,cache_size=1000).fit(X_res,y_res)
            else:
                est = SVC(C=4.25,kernel='rbf',gamma=0.0001,shrinking=True, \
                           cache_size=1000,class_weight='balanced').fit(X_res,y_res)
                
            self.estimators_ += [est]
            if self.verbose:
                print(i+1,',',end='')
        print('')
        return self
    
    def most_common_classes(self, lst):
        # Given: array of arrays of predictions for each estimators
        lst = np.transpose(np.array(lst)) # we want array of arrays of predictions for each instance in X
        data = [Counter(sub_lst) for sub_lst in lst]
        # Get the most common class predictions (could be multiple)
        mc_dup = [ [(x[0],x[1]) for x in sub_data.most_common() if sub_data.most_common()[0][1] == x[1]] for sub_data in data]
        # If multiple most common class predictions, choose random (if len(x)=1 then we chose randint(0,0)=0)
        mc = [x[randint(0,len(x)-1)][0] for x in mc_dup]
        return mc
    
    def predict(self,X):
        if self.verbose:
            print("Predicting with ", self.n_clf, "estimators")
        l = len(self.estimators_)
        pred1 = []
        for i in range(l): # Used loop rather than list comprehension to simplify verbosity
            pred1 += [self.estimators_[i].predict(X)]
            print(i+1,',',end='')
        print('')
        pred = self.most_common_classes(pred1)
        return pred