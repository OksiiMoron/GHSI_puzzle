import numpy as np

def pcs_threshold(explained, threshold):
    """Gives back the number of principal components that hit the given threshold

    Args:
        explained : Gives the % of explained variance that the certain PC covers
        threshold : Threshold that we set; Number between 1 and 100.
    """

    cumulative_sum = np.cumsum(explained)
    nPCs = np.searchsorted(cumulative_sum, threshold) + 1

    # Let\s say that our first 3 components give a cumsum of 85. Because our third
    # component is on index 2 (which we are returning), but we need the number 
    # of components that cummulatively give meet the threshold, we add + 1.
    return nPCs