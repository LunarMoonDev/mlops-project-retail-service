from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from config import conf


class ColumnTransformerWrapper(BaseEstimator, TransformerMixin):
    """wrapper class for ColumnTransformer
    - renames the columns from transformed result
    """

    def __init__(self, transformer):
        self.transformer = ColumnTransformer(
            transformers=[("scaler", transformer, conf.FEATURES_SCALE_LIST)],
            remainder="passthrough",
        )

        self.transformer.set_output(transform="pandas")

    def fit(self, df: DataFrame, y=None):
        """Fits ColumnTransformer

        Args:
            df (DataFrame): given input
            y (_type_, optional): y value set (Default: None)

        Returns:
            _type_: instance of ColumnTransformerWrapper
        """
        self.transformer.fit(df, y)
        return self

    def transform(self, df) -> DataFrame:
        """transforms with column transformer and rename the result

        Args:
            df (_type_): given input

        Returns:
            DataFrame: transformed dataframe with renamed cols
        """
        df_copy = self.transformer.transform(df)

        df_copy.columns = [col.removeprefix("scaler__") for col in df_copy.columns]
        df_copy.columns = [col.removeprefix("remainder__") for col in df_copy.columns]

        return df_copy
