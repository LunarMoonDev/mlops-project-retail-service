from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from config import conf
from functools import partial

class ImputateTransformer(FunctionTransformer):
    """
        Transformer class for imputating values
    """
    def fit(self, X: DataFrame, y=None) -> 'ImputateTransformer':
        """
            fits the data to make an imputate look up table
        """

        aggre_df = X.copy().dropna()
        aggre_df = aggre_df[[conf.DATA_IMPUTATE_CATEGORY] + conf.DATA_IMPUTATE_FEATURES]
        aggre_func = dict(zip(conf.DATA_IMPUTATE_FEATURES, conf.DATA_IMPUTATE_STRAT))
        self.look_up = aggre_df.groupby(conf.DATA_IMPUTATE_CATEGORY).agg(**aggre_func)

        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        """
            imputates the dataframe on missing values
        """
        X_imputed = X.copy()

        categories = X_imputed[conf.DATA_IMPUTATE_CATEGORY]
        for feature in conf.DATA_IMPUTATE_FEATURES:
            map_func = partial(self.__grab_value, y=feature)
            X_imputed[feature] =  X_imputed[feature].fillna(categories.map(map_func))

        return X_imputed
    

    def __grab_value(self, feature: str, item: str):
        """
            grabs the imputes value
        """
        mask = self.look_up[conf.DATA_IMPUTATE_CATEGORY] == item
        return self.look_up[mask][feature]
        