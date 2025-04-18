import pandas as pd

from config import conf

def pre_process(data: pd.DataFrame, target: bool = False):
    """
        This function will perform the following for cleanup:
            - fix data types of features
            - return X which is the feature for the prediction
            - return Y which is what we want to predict
    """
    data[conf.CAT_FEATURES] = data[conf.CAT_FEATURES].astype(str)
    data[conf.NUM_FEATURES] = data[conf.NUM_FEATURES].astype(float)
    data[conf.DATE_FEATURES] = pd.to_datetime(data[conf.DATE_FEATURES], dayfirst=True)

    X = data[conf.FEATURES]
    if target:
        Y = data[conf.TARGET]
        return X, Y
    
    return X
