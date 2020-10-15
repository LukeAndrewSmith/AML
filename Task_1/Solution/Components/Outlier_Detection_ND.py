import pandas as pd
import numpy as np

def knn(x):
    # x=...
    # return x
    print("Hi")

def _get_cov(X):
    # calculates cov matrix
    from sklearn.covariance import EmpiricalCovariance
    cov = EmpiricalCovariance().fit(X)
    return cov

def magic_indices(X_train,y_train, n_outliers=10, mask=None, verbose=False):
    # gained from loocv 
    # 180 seems to be best for lasso 
    magic_indices = [323, 333, 215, 1094, 1179, 774, 527, 826, 981, 673, 313, 765, 
                    1022, 487, 627, 657, 1199, 431, 42, 789, 240, 923, 989, 890, 988, 
                    317, 1182, 16, 412, 809, 120, 62, 357, 1078, 1000, 136, 818, 571, 
                    344, 34, 942, 1189, 845, 1042, 204, 537, 960, 1201, 390, 908, 990, 
                    611, 811, 880, 875, 1021, 1205, 968, 884, 270, 1115, 639, 200, 1067, 
                    1117, 1070, 1130, 947, 559, 1098, 920, 927, 854, 1195, 1100, 900, 1157, 
                    899, 1036, 869, 384, 251, 911, 1135, 228, 1009, 1209, 1128, 1197, 733, 
                    1063, 859, 905, 459, 146, 110, 1113, 1096, 1093, 1162, 889, 821, 7, 993, 
                    1047, 962, 914, 812, 372, 1178, 1148, 903, 1172, 820, 1198, 677, 1003, 
                    1069, 825, 1210, 1145, 1028, 340, 1065, 46, 936, 800, 860, 1134, 898, 
                    616, 628, 1046, 810, 1207, 963, 1077, 1196, 1177, 984, 1165, 217, 885, 9,
                    50, 1048, 1141, 1138, 964, 1149, 1171, 819, 965, 133, 206, 1083, 866, 852,
                    895, 1183, 1176, 907, 202, 1005, 153, 1109, 670, 1010, 874, 856, 1190, 1091,
                    967, 233, 1184, 1033, 814, 886, 179, 385, 1150, 904, 992, 1044, 980, 1079, 
                    1081, 1020, 983, 1159, 994, 35, 896, 1174, 282, 1106, 901, 867, 827, 954, 
                    939, 834, 871, 970, 858, 1049, 822, 1122, 1200, 1180, 1019, 1034, 958, 969, 
                    868, 892, 974, 1131, 1187, 1076, 1105, 953, 1164, 955, 979, 833, 1025, 816, 
                    973, 1158, 951, 1055, 862, 181, 1037, 1040, 1124, 361, 1206, 1006, 1032, 815, 
                    1154, 1168, 878, 319, 131, 891, 837, 1085, 597]

    # do nothing if no lines selected
    if n_outliers == 0:
        return X,y,mask
    
    # take first n_outliers
    selected = magic_indices[0:n_outliers]
    outliers = np.asarray(selected)
    if verbose:
        print('# Outliers: ', outliers)
        
    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])

    if mask is None:
        return X,y
    else:
        mask = pd.DataFrame(mask).drop(y_train.index[outliers]).to_numpy()
        return X,y,mask


def mahalanobis_distance(X_train, y_train, mask='', alpha=3, verbose=False):
    #join x train and y train
    train_all = X_train.copy()
    train_all['age'] = y_train
    numpy_all = train_all.to_numpy()
    
    # calc cov matric
    cov = _get_cov(numpy_all)
    mahal = cov.mahalanobis(train_all)
    mahal_mean = np.mean(mahal)
    mahal_std = np.std(mahal)

    # get all id that have higher z score than alpha and remove them
    outliers = np.where((np.asarray(mahal) - mahal_mean) / mahal_std > alpha)
    if verbose:
        print('# Outliers: ', outliers)
    
    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])
    if mask == '':
        return X,y
    else:
        mask = pd.DataFrame(mask).drop(y_train.index[outliers]).to_numpy()
        return X,y,mask


def dbscan(X_train, y_train, mask='', verbose=False):
    from sklearn.cluster import DBSCAN
    train_all = X_train.copy()
    train_all['age'] = y_train

    clustering = DBSCAN(eps=42, min_samples=2).fit(train_all)
    labels = clustering.labels_
    if verbose:
        print(labels)

    outliers = np.where((np.asarray(labels) == -1))
    if verbose:
        print('# Outliers: ', outliers)

    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])

    if mask == '':
        return X,y
    else:
        mask = pd.DataFrame(mask).drop(y_train.index[outliers]).to_numpy()
        return X,y,mask


def isolation_forest(X_train, y_train, mask='', verbose=False):
    from sklearn.ensemble import IsolationForest
    train_all = X_train.copy()
    train_all['age'] = y_train

    clf = IsolationForest(n_estimators=10, warm_start=True)
    clf.fit(train_all)  # fit 10 trees  
    clf.set_params(n_estimators=20)  # add 10 more trees  
    clf.fit(train_all)
    labels = clf.predict(train_all)
    outliers = np.where(np.asarray(labels)==-1)
    if verbose:
        print('# Outliers: ', outliers)

    X = X_train.drop(X_train.index[outliers])
    y = y_train.drop(y_train.index[outliers])
    if mask == '':
        return X,y
    else:
        mask = pd.DataFrame(mask).drop(y_train.index[outliers]).to_numpy()
        return X,y,mask
   