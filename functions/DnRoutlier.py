import numpy as np

def DnRoutlier(data):
    """
    Replacing outliers from a numpy matrix with the median for that columns.
    """
    data_clean = data.copy()
    
    
    for i in range(data.shape[1]):
        col = data[:, i]
        Q1, Q3 = np.percentile(col, [25, 75])
        IQR = Q3 - Q1
        median_val = np.median(col)
        
        
        outlier_mask = (col < Q1 - 1.5 * IQR) | (col > Q3 + 1.5 * IQR)
        
        
        col[outlier_mask] = median_val
        
        data_clean[:, i] = col
        
    return data_clean
