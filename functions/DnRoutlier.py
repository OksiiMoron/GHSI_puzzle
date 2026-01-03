import numpy as np
from scipy.stats import median_abs_deviation

def DnRoutlier(data):
    data_clean = data.copy()

    for i in range(data.shape[1]):
        col = data_clean[:, i]

        med = np.nanmedian(col)
        mad = median_abs_deviation(col, nan_policy='omit')

        if mad == 0 or np.isnan(mad):
            continue

        threshold = 3 * mad  
        outliers = np.abs(col - med) > threshold

        col[outliers] = med
        data_clean[:, i] = col

    return data_clean
