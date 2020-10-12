def z_score(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    z_scores = X.apply(lambda row: (row-mean).div(std), axis=1)
    threashold = 3
    return X.mask(z_scores>threashold)
