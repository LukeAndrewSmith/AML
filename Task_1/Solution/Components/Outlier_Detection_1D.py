
def z_score(X,x_extra=None,verbose=False):
    if x_extra is not None:
        X = X.append(x_extra)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    if x_extra is not None:
        n_extra_rows = len(x_extra.index)
        X = X[:-n_extra_rows]
    z_scores = X.apply(lambda row: (row-mean).div(std), axis=1)
    threashold = 3
    if verbose:
        print('# Outliers: ', (z_scores>threashold).sum().sum())
    return X.mask(z_scores>threashold)

class Z_Score_Outlier():
    def __init__(self, threashold=3):
        self.threashold = threashold

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.z_scores = X.apply(lambda row: (row-mean).div(std), axis=1)
        return self

    def transform(self, X, y=None):
        try:
            return X.mask(z_scores>threashold)
        except Exception as err:
            print('Z_Score_Outlier.transform(): {}'.format(err))
        return X

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)

    def get_params(self, deep=True):
        return {"threashold": self.threashold}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self