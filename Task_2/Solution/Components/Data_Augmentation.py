
import imblearn
from imblearn.over_sampling import SMOTE

def smote_resampling(X,y):
    sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X,y)

    return X_res, y_res