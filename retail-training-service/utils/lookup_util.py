from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class ImputateTransformer(BaseEstimator, TransformerMixin):
    """Transformer class for imputating values"""

    def __init__(
        self, by: str = None, columns: list[str] = None, strats: list[str] = None
    ):
        self.by = by
        self.columns = columns
        self.strats = strats

    # pylint: disable=unused-argument
    def fit(self, df: DataFrame, y=None):
        """fits the data to make an imputate look up table

        Args:
            df (DataFrame): given input dataframe
            y (_type_, optional): y values (Default: None)

        Returns:
            _type_: instance of ImputateTransformer
        """

        aggre_df = df.copy().dropna()
        aggre_df = aggre_df[[self.by] + self.columns]

        # converting dict to tuple
        aggregations = zip(self.columns, self.strats)
        aggre_dict = {}

        # for naming scheme
        for key, aggregate in zip(self.columns, aggregations):
            aggre_dict[key] = aggregate

        # apparently merge still works even if self.by is index, maybe faster?
        lookup_df = aggre_df.groupby(self.by).agg(**aggre_dict)

        # pylint: disable=attribute-defined-outside-init
        self.lookup_df = lookup_df
        # pylint: enable=attribute-defined-outside-init
        return self

    # pylint: enable=unused-argument

    def transform(self, df: DataFrame) -> DataFrame:
        """imputates the dataframe on missing values

        Args:
            df (DataFrame): given dataframe

        Returns:
            DataFrame: imputated dataframe
        """

        imputated_df = df.copy()
        imputated_df_filled = df.merge(
            self.lookup_df, on=self.by, how="left", suffixes=("", "_lookup")
        )
        lookup_cols = [f"{col}_lookup" for col in self.columns]

        for col, lookup_col in zip(self.columns, lookup_cols):
            imputated_df_filled[col] = imputated_df_filled[col].combine_first(
                imputated_df_filled[lookup_col]
            )

        imputated_df = imputated_df_filled.drop(columns=lookup_cols)

        return imputated_df
