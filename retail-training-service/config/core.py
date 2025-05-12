from pathlib import Path

from pydantic import BaseModel
from strictyaml import load

# Project directories
PACKAGE_ROOT = Path(__file__).resolve().parents[0]
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yaml"


class Config(BaseModel):
    """class for the training project config"""

    FEATURE_TARGET: str  # TARGET Y OF DATA
    FEATURES_LIST: list[str]  # LIST OF FEATURES
    FEATURES_SCALE_LIST: list[str]  # LIST OF FEATURES TO SCALE
    FEATURES_MAPPING: dict[str, str]  # MAPPING TO RENAME FEATURES
    FEATURES_DATE: list[str]  # DATE FEATURES
    NUMERICALS: list[str]  # NUMERICAL FEATURES
    CATEGORICALS: list[str]  # CATEGORICAL FEATUREES

    FILTER_TOGGLE: bool
    DROP_TOGGLE: bool

    FILTER_FEATURE: str  # FEATURE TO FILTER
    FILTER_MAX: int  # MAX OF FILTER
    FILTER_MIN: int  # MIN OF FILTER

    RANDOM_STATE: int

    DATA_IMPUTATE_CATEGORY: str  # ITEM PURCHASED
    DATA_IMPUTATE_FEATURES: list[str]  # FEATURES TO IMPUTATE
    DATA_IMPUTATE_STRAT: list[str]  # STRAT OF IMPUTATION

    BATCH_TODAY_OFFSET: int  # OFFSET FROM TODAY
    BATCH_DATA_PATH: str  # PATH TO BATCH DATA
    BATCH_TEST_SIZE: float

    TRACKING_URI: str
    EXPERIMENT_NAME: str

    XG_MAX_DEPTH_MIN: int
    XG_MAX_DEPTH_MAX: int
    XG_N_ESTIMATORS_MIN: int
    XG_N_ESTIMATORS_MAX: int
    XG_N_ESTIMATORS_STEP: int
    XG_N_LEARNING_RATE_MIN: float
    XG_N_LEARNING_RATE_MAX: float
    XG_CHILD_WEIGHT_MIN: float
    XG_CHILD_WEIGHT_MAX: float
    XG_EARLY_STOPPING_ROUNDS: int
    XG_N_TRIALS: int


with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
    config_file = f.read()

conf = Config(**load(config_file).data)
