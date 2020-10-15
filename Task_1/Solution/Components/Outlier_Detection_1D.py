
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
    def __init__(self, percentiles=3):
        self.percentiles = percentiles

    def fit(self, X, Y):
       return self

    def transform(self, X, Y=None):
        try:
            if self.features < X.shape[1]:
                if Y is not None:
                    self.selector.fit(X, Y)
                return selector.transform(X)
        except Exception as err:
            print('MyFeatureSelector.transform(): {}'.format(err))
       return X

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)