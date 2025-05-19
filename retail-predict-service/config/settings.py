from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parents[0]
ROOT = PACKAGE_ROOT.parent
DOT_ENV_FILE = str(ROOT / ".env")


class FeatureColumns(BaseModel):
    """describes the map of batch data column into cols specified below"""

    CustomerId: str = 'CustomerId'
    ItemPurchased: str = 'ItemPurchased'
    PurchaseAmount: str = 'PurchaseAmount'
    DatePurchase: str = 'DatePurchase'
    ReviewRating: str = 'ReviewRating'
    PaymentMethod: str = 'PaymentMethod'
    OrderId: str = 'OrderId'


class Settings(BaseSettings):
    """Config class from env vars"""

    model_config = SettingsConfigDict(env_file=DOT_ENV_FILE, env_file_encoding='utf-8')

    INPUT_BATCH_PATH: Annotated[
        str, Field(..., description='path to s3 bucket containing batch data')
    ]
    FEATURES_MAPPING: Annotated[
        FeatureColumns,
        Field(
            default_factory=FeatureColumns,
            description='mapping for renaming the columns',
        ),
    ]

    INPUT_DAY_OFFSET: int = Field(
        default=0, description='days to offset batch data date'
    )
    INPUT_READ_THREADS: int = Field(
        default=2, description='threads used to read the csv files'
    )

    FEATURES_CAT: list[str] = Field(
        ..., description='set of columns tagged as categorical'
    )
    FEATURES_NUM: list[str] = Field(
        ..., description='set of columns tagged as numericals'
    )
    FEATURES_DATE: list[str] = Field(
        ..., description='set of columns tagged as date type'
    )
    FEATURES_LIST: list[str] = Field(..., description='set of feature columns')

    REGISTRY_TRACKING_URI: str = Field(..., description='Host to mlflow server')
    REGISTRY_MODEL_NAME: str = Field(
        default='retail_review_prediction', description='model name for our service'
    )
    REGISTRY_ALIAS_NAME: str = Field(
        default='champion', description='alias associated for the model to run'
    )

    DB_NAME: str = Field(..., description='name of database')
    DB_USER: str = Field(..., description='username for interacting with db')
    DB_PASSWORD: str = Field(..., description='password for interacting with db')
    DB_HOST: str = Field(..., description='host of database server')
    DB_PORT: int = Field(..., description='port of database server')
