import numpy as np

def durbin_watson(res, y):
    """
    Calculates the Durbin Watson statistic. Its sorted by Price to see if there is autocorrelation
    """
    sort_index = np.argsort(y,axis=0)
    res_ = res.copy()
    res_ = res_.iloc[sort_index]
    res_sum_ = res_ - np.roll(res_, 1)
    res_sum_top = np.sum(res_sum_[1:]**2)
    res_sum_bot = np.sum(res**2)
    return res_sum_top / res_sum_bot

def mean_absolute_percentage_error(y_pred, y_true):
    """
    Calculates the mean absolute percentage error of two arrays
    """
    n = len(y_true)
    return np.sum(np.abs((y_pred - y_true)) / y_true) / n 

def negative_mean_absolute_percentage_error(y_pred, y_true):
    """
    Calculates the negative mean absolute percentage error of two arrays
    """
    return - mean_absolute_percentage_error(y_pred, y_true)



def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    https://stackoverflow.com/questions/46498455/categorical-features-correlation
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

