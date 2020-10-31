import numpy as np

def transform(X,y):
    # generate both datasets needed
    X_1 = X.copy()
    X_2 = X.copy()
    y_1 = y.copy()
    y_2 = y.copy()

    # for x_1 , y_1 convert all y==2 to y==0, no change for x_1 needed
    y_1['y'] = np.where((y_1.y == 2), 0, y_1.y)
    

    # for x_2, y_2 delete all rows where y==1
    X_2 = X_2[y['y'] != 1]
    y_2 = y_2[y['y'] != 1]

    return X_1, y_1, X_2, y_2

def combine_predictions(y_pred_1, y_pred_2):
    y_pred = list()
    for i, y in enumerate(y_pred_1):
        # take the prediction of y_pred_1 if y==1 (large group), else need to check the distinguished prediction from y_pred_2
        if y == 1:
            y_pred.append(y)
        else:
            if y_pred_2[i] == 0:
                y_pred.append(0)
            elif y_pred_2[i] == 2:
                y_pred.append(2)
            
    return y_pred

def vote_prediction(y_pred_1, y_pred_2, y_pred_3):
    # take the majority vote
    y_pred = list()
    for i, y in enumerate(y_pred_1):
        # take the prediction of y_pred_1 if y==1 (large group), else need to check the distinguished prediction from y_pred_2
        if y == y_pred_2[i] or y==y_pred_3[i]:
            y_pred.append(y)
        else:
            # the other two have to be the same then
            y_pred.append(y_pred_2[i])
            
    return y_pred

