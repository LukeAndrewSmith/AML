def z_score(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z_scores = x.apply(lambda row: (row-mean).div(std), axis=1)
    threashold = 3
    return x.mask(z_scores>threashold)