from sklearn.feature_selection import SelectKBest, chi2, f_classif

def selectkbest(X,y, X_test, k_best=50):
    selector = SelectKBest(f_classif, k=k_best).fit(X, y)
    selector.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:,cols]
    X_test_new = X_test.iloc[:,cols]

    return X_new, X_test_new
    
    