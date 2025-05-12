from functools import partial

from typing import Union, Dict
import pandas as pd
from config import conf
from toolz import compose_left
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
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
    return df[conf.FEATURES]

def convert_to_dataframe(data: Union[pd.DataFrame, Dict]):
    """
        pipeline to convert given input data as dataframe
        ensures downstream goes well
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    else:
        raise TypeError(f'Expected dataframe or Dict, got {type(data).__name__}')

def convert_to_dict(data: Union[pd.DataFrame, Dict]) -> Dict:
    """
        pipeline to convert data into dict
    """
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError(f'Expected dataframe or Dict, got {type(data).__name__}')

def preprocessor_with_transform(df: pd.DataFrame, transform: type = ()):
    """
        creates a one big function of function ception with transform
    """
    return compose_left(transform)(df)

def feature_pipeline(transforms: tuple = ()) -> Pipeline:
    """
        creates a feature pipeline that expects dataframe
        processes the dataframe with bunch of transformers
        then expects a dictionary from the output transformers
        then outputs a DictVectorizer for the model


        transformers can be
            - feature_selector
            - ImputateTransformer
            - PowerTransformer for scaling
    """
    pre_processor = partial(preprocessor_with_transform, transform=transforms)
    
    return make_pipeline(
        FunctionTransformer(convert_to_dataframe),
        *transfomers,
        FunctionTransformer(convert_to_dict),
        DictVectorizer()
    )

def make_model_pipeline(pipeline: Pipeline, model: BaseEstimator) -> Pipeline:
    """
        adds the estimator at the end of the pipeline :)
    """
    return make_pipeline(pipeline, model)