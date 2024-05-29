# Orchestration

## Pre-requisites

1. Clone the pre-made Mage data pipeline repository
```bash
git clone https://github.com/mage-ai/mlops.git
```

2. Move into the cloned repository and run the .sh script launching Mage and PostrgeSQL
```bash
cd mlops
. /scripts/start.sh
```

## Introduction

### MLOps involves 4 steps

1. Preparing the modelf or deployment involves optimizing performance, ensuring it handles real-world data, and packaging it for integration into existing systems.

2. Deploying the model involves moving it from development to production, making it accessible to users and applications.

3. Once deployed, models must be continuously monitored for accuracy and reliability, and may need retraining on new data and updates to maintain effectiveness.

4. The operationalized models must be integrasted into existing workflows, applications, and decision-making processes to drive business impact.

### Why do we need to "operationalize" ML?

1. Productivity

    MLOps allows for easier collaboration between those working on ML including data scientists, ML engineers, and DevOps teams by providing a unified environment for experiment tracking, feature engineering, model management, and deployment. This helps break down silos and facilitates the machien learning lifecycle.

2. Reliability

    MLOps ensures high-quality, reliable models in production through clean datasets, proper stesting, validation, CD/CD practices, monitoring, and governance.

3. Reproducibility

    MLOps enables reproducibility and compliance by versioning datasets, code, and models, providing transparency and auditability to ensure adherence to policies and regulations.

4. Time-to-value

    MLOps streamlines the ML lifecycle, enabling organizations to successfully deploy more projects to production and derive tangible business value and ROI from AI/ML investments at scale.

## Mage

### Introduction

We are using Mage, a data orchestration tool, to help enable steps of the MLOps lifecycle.

1. Data preparation

    Mage offers features to build, run, and manage data pipelines for data transformation and integration, including pipeline orchestration, notebook environments, data integrations, and strreaming pipelines for real-time data.

2. Training and deployment

    Mage helps prepare data, train machine learning models, and deploy them with accessible API endpoints.

3. Standardize complex processes

    Mage simplifies MLOps by providing a unified platform for data pipelining, model development, deployment, versioning, CI/CD, and maintenance, allowing developers to focus on model creation while improving efficiency and collaboration.

If you followed the pre-requisites Mage should be running and can be accessed via the default port 6789: 

`http://localhost:6789`

Navigate to the pipelines and run `example_pipeline`

### Mage Basics

#### Creating a new project and pipeline

Open your Mage UI

1. Open `text editor`
2. Right click directory and select `New Mage project` and name it i.e. `unit_1_data_preparation`
3. Register the project in `Settings` and select as `Currently selected project` 
4. Switch to the project in the top right i.e. ` mlops > unit_1_data_preparation`
5. In the `Overview` tab Select `New pipeline` in this case we name it `Data preparation`

#### Creating ingest block

Make sure you are in the previously created pipeline

1. Select `All blocks` then `Data loader` then `Base tepmlate` in this case we name it `ingest`

2. Add desired code

Adding the following code:
```python
import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2024, (1, 3))]:
        for i in range(*months):
            response = requests.get(
                'https://github.com/mage-ai/datasets/raw/master/taxi/green'
                f'/{year}/{i:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(response.text)

            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)

    return pd.concat(dfs)
```

3. Explore the data by selecting the `Charts` button ont he top right of the block.

#### Creating utility functions

1. Open `text editor` and create `New file` or `New folder` in our case we create both at once via `New file` named `utils/data_preparation/cleaning.py`
    - `utils/data_preparation/cleaning.py`
    - `utils/data_preparation/feature_engineering`
    - `utils/data_preparation/feature_selector`
    - `utils/data_preparation/splitters`
    - `utils/data_preparation/__init__.py`

In our case we are using the below code for each function.

cleaning.py
```python
import pandas as pd


def clean(
    df: pd.DataFrame,
    include_extreme_durations: bool = False,
) -> pd.DataFrame:
    # Convert pickup and dropoff datetime columns to datetime type
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    # Calculate the trip duration in minutes
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    if not include_extreme_durations:
        # Filter out trips that are less than 1 minute or more than 60 minutes
        df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert location IDs to string to treat them as categorical features
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df
```

feature_engineering.py
```python
from typing import Dict, List, Union

from pandas import DataFrame


def combine_features(df: Union[List[Dict], DataFrame]) -> Union[List[Dict], DataFrame]:
    if isinstance(df, DataFrame):
        df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    elif isinstance(df, list) and len(df) >= 1 and isinstance(df[0], dict):
        arr = []
        for row in df:
            row['PU_DO'] = str(row['PULocationID']) + '_' + str(row['DOLocationID'])
            arr.append(row)
        return arr
    return df
```

feature_selector.py
```python
from typing import List, Optional

import pandas as pd

CATEGORICAL_FEATURES = ['PU_DO']
NUMERICAL_FEATURES = ['trip_distance']


def select_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    if features:
        columns += features

    return df[columns]
```

splitters.py
```python
from typing import List, Tuple, Union

from pandas import DataFrame, Index


def split_on_value(
    df: DataFrame,
    feature: str,
    value: Union[float, int, str],
    drop_feature: bool = True,
    return_indexes: bool = False,
) -> Union[Tuple[DataFrame, DataFrame], Tuple[Index, Index]]:
    df_train = df[df[feature] < value]
    df_val = df[df[feature] >= value]

    if return_indexes:
        return df_train.index, df_val.index

    if drop_feature:
        df_train = df_train.drop(columns=[feature])
        df_val = df_val.drop(columns=[feature])

    return df_train, df_val
```

#### Adding block to utilize utility functions

1. Navigate to the pipeline we created `unit_1_data_preparation/pipelines/data_preparation` and enter `edit` mode.
2. Scroll down and select `All blocks` then `Transformer` then `Base template` and name it `preparing`.
3. Add your code.

preparing.py:
```python
from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_on_feature = kwargs.get('split_on_feature')
    split_on_feature_value = kwargs.get('split_on_feature_value')
    target = kwargs.get('target')

    df = clean(df)
    df = combine_features(df)
    df = select_features(df, features=[split_on_feature, target])

    df_train, df_val = split_on_value(
        df,
        split_on_feature,
        split_on_feature_value,
    )

    return df, df_train, df_val
```

4. Notice in our code we use variables `split_on_feature`, `split_on_feature_value`, and `target` to make it reusable. You will want to set the default values in the `Global variables` settings which can be accessed on the panel on the right.
    - `split_on_feature`: lpep_pickup_datetime
    - `split_on_feature_value`: 2024-02-01
    - `target`: duration
5. Run the pipeline

#### Adding encoders

1. create file `utils/data_preparation/encoders.py`

encoders.py
```python
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer


def vectorize_features(
    training_set: pd.DataFrame,
    validation_set: Optional[pd.DataFrame] = None,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, DictVectorizer]:
    dv = DictVectorizer()

    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    X_val = None
    if validation_set is not None:
        val_dicts = validation_set[training_set.columns].to_dict(orient='records')
        X_val = dv.transform(val_dicts)

    return X_train, X_val, dv

```

#### Creating build block

1. Add a 3rd step to our pipeline `unit_1_data_preparation/pipelines/data_preparation/` by selecting `Data exporter` then `Base template` and name is `build` in our case.
2. Add your code

build.py
```python
from typing import List, Tuple

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv


@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X.shape[0] == 105870
    ), f'Entire dataset should have 105870 examples, but has {X.shape[0]}'
    assert (
        X.shape[1] == 7027
    ), f'Entire dataset should have 7027 features, but has {X.shape[1]}'
    assert (
        len(y.index) == X.shape[0]
    ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'


@test
def test_training_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X_train.shape[0] == 54378
    ), f'Training set for training model should have 54378 examples, but has {X_train.shape[0]}'
    assert (
        X_train.shape[1] == 5094
    ), f'Training set for training model should have 5094 features, but has {X_train.shape[1]}'
    assert (
        len(y_train.index) == X_train.shape[0]
    ), f'Training set for training model should have {X_train.shape[0]} examples, but has {len(y_train.index)}'


@test
def test_validation_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X_val.shape[0] == 51492
    ), f'Training set for validation should have 51492 examples, but has {X_val.shape[0]}'
    assert (
        X_val.shape[1] == 5094
    ), f'Training set for validation should have 5094 features, but has {X_val.shape[1]}'
    assert (
        len(y_val.index) == X_val.shape[0]
    ), f'Training set for training model should have {X_val.shape[0]} examples, but has {len(y_val.index)}'
```

Note we also added test code in this block where you will see `X/X tests passed.` upon running.