import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_df_hist(X,n_features=20,n_bins=10):
    to_plot = X.loc[:,0:n_features]
    to_plot.hist(bins=n_bins)
    plt.show()