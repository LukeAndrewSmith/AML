# Warning !!! when changing this file, you must do the same reloading as we did with other pipeline components in deepnote

class Preprocessing():
    a = "local state"

    def fit(self, X, y=None):
        self.a = X
        return self
    
    def transform(self, X=None, y=None, action="scaler"):
        if (action == "scaler"):
            return X
        elif (action == "something"):
            return X
        else:
            return X