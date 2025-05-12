from typing import Dict, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from config import conf


def feature_selector(df: pd.DataFrame) -> pd.DataFrame:
    """pipeline to grab features from given df

    Args:
        df (DataFrame): given input

    Returns:
        DataFrame: feature dataframe
    """
    return df[conf.FEATURES_LIST]


def convert_to_dataframe(data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
    """pipeline to convert given input data as dataframe
        - ensures downstream goes well

    Args:
        data (Union[pd.DataFrame, Dict]): given input

    Raises:
        TypeError: error for wrong type

    Returns:
        DataFrame: input as df
    """

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame(data)

    raise TypeError(f"Expected dataframe or Dict, got {type(data).__name__}")


def convert_to_dict(data: Union[pd.DataFrame, Dict]) -> Dict:
    """pipeline to convert data into dict

    Args:
        data (Union[pd.DataFrame, Dict]): given input

    Raises:
        TypeError: _deserror for wrong typecription_

    Returns:
        Dict: input as dict
    """

    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, dict):
        return data

    raise TypeError(f"Expected dataframe or Dict, got {type(data).__name__}")


def feature_pipeline(transfomers: tuple = ()) -> Pipeline:
    """pipeline that performs the following in succession:
        - convert_to_dataframe
        - transfomers
        - convert_to_dict
        - DictVectorizer

        transformers can be
            - ImputateTransformer
            - PowerTransformer

    Args:
        transfomers (tuple, optional): _description_. transformers to apply in pipeline.

    Returns:
        Pipeline: feature pipeline
    """
    return make_pipeline(
        FunctionTransformer(convert_to_dataframe),
        *transfomers,
        FunctionTransformer(convert_to_dict),
        DictVectorizer(),
    )


def make_model_pipeline(pipeline: Pipeline, model: BaseEstimator) -> Pipeline:
    """adds the model at the end of the pipeline

    Args:
        pipeline (Pipeline): feature pipeline
        model (BaseEstimator): model trained

    Returns:
        Pipeline: model pipeline
    """
    return make_pipeline(pipeline, model)
