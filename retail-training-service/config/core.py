from pathlib import Path

from pydantic import BaseModel
from strictyaml import load

# Project directories
PACKAGE_ROOT = Path(__file__).resolve().parents[0]
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / 'properties' / 'config.yaml'

class Config(BaseModel):
    """
        class for the training project config
    """
    TARGET: str

    FILTER_FEATURE: str
    FILTER_MAX: int
    FILTER_MIN: int

    RANDOM_STATE: int

    FEATURES: list[str]
    NUM_FEATURES: list[str]
    CAT_FEATURES: list[str]
    DATE_FEATURES: list[str]

    DATA_IMPUTATE_CATEGORY: str
    DATA_IMPUTATE_FEATURES: list[str]
    DATA_IMPUTATE_STRAT: list[str]

    SCALE_FEATURES: list[str]

    TRAIN_SAMPLE: str
    VALID_SAMPLE: str
    MODEL_SAMPLE: str

with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
    config_file = f.read()

conf = Config(**load(config_file).data)
