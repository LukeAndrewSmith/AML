def z_score(X,verbose=False):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    z_scores = X.apply(lambda row: (row-mean).div(std), axis=1)
    threashold = 3
    if verbose:
        print('# Outliers: ', (z_scores>threashold).sum().sum())
    return X.mask(z_scores>threashold)
