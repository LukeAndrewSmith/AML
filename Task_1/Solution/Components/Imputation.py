from sklearn.impute import SimpleImputer
import np

def median(x):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
    return pd.DataFrame(fill_NaN.fit_transform(x))

def impute():
    print("Hi")