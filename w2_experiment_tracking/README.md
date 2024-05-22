# Experiment Tracking

## Introduction to Experiment Tracking

Experiment tracking is the process of keeping track of all relevant information from an ML experiment:
- source code
- environment
- data
- model
- hyperparameters
- metrics

**MLflow** is an open source platform for the machine learning lifecycle which includes experiment tracking. It is a python package that can be installed with pip and contains four main modules:
- tracking
- models
- model registry
- projects

**tracking** module allow you to organize your experiment into "runs" which keep track of:
- parameters (such as preprocessing of data)
- metrics
- metadata
- artifacts
- models

MLflow also automatically logs
- source code
- version of the code (git commit)
- start and end time
- author

## MLflow

### Installation

Because all of the packages we need, mainly `mlflow` it is best practice to create a conda environment.

Create conda environment:
```bash
conda create -n ml-ops-zoomcamp-01 python=3.9
```

Activate conda environment:
```bash
conda activate ml-ops-zoomcamp-01
```

We created a package list in w2_experimenttracking/requirements_w2.txt:
```bash
pip install -r requirements_w2.txt
```

Ensure packages were installed
```bash
pip list
```

### Introduction

You can run mlflow GUI with this simple command:
```bash
mlflow ui
```

You will encounter errors though for some operations and be limited unless you specify a backend database. This command will help with that:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
*Note you can leave out `--port 5001` I am doing this because I am currently using port 5000.*

We have copied the data from `data/green_tripdata_*` as well as the notebook from last week `notebooks/w1_duration-prediction.ipynb`.

Next we will modify the copied notebook to add mlflow.

These lines were added to add this model to mlflow:
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db") #specify the backend you're using sqlite in this case
mlflow.set_experiment("nyc-taxi-experiment") #experiment name, if it doesn't exist it will create it
```

After running that code you can see in the GUI that was opened previously that the new experiment is listed.

Next we want to use mlflow to track experiments.

Here is the original python for the linear regression:
```python
lr = Ridge(alpha = 0.1) # Note I tested multiple alpha values
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)
```

Here we use mlflow to track experiments:
```python
# Let's try another model Ridge regression another form of regulrizated linear regression
with mlflow.start_run():

    # mlflow.set_tag("tag1", "tag2") # You can use this to add developer name and other useful tags
    # Here is an example of parameter logging where we do basic data "versioning"
    mlflow.log_param("train-data-path", "../data/green_tripdata_2021-01.csv")
    mlflow.log_param("val-data-path", "../data/green_tripdata_2021-02.csv")

    alpha = 0.01
    mlflow.log_param("alpha", alpha) # Keep track of each alpha we pass

    lr = Ridge(alpha) # Note I tested multiple alpha values
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
```

The new notebook can be found [here](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w1_intro/notebooks/w2_duration-prediction.ipynb).

### Experiment Tracking with MLflow

We will be adding an additional experiment to the [notebook](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w2_experiment_tracking/notebooks/w2_duration-prediction.ipynb). Key portions of the notebook will be listed below as they are added to the notebook.

This is the code that was added to the notebook with notes:
```python
# Experiment tracking demo

import xgboost as xgb

"""
fmin - finds minimum of output
tpe - algorithm to control logic
hp - library that contains methods to define ranges for each hyperparameter
STATUS_OK - at end of each run tells the system it is ok
Trials - keeps track of information from each run
scope - determines range of type int
"""
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# First we must set the train and validation data to the correct format for xgboost
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params, # Specify parameters which are input variables in this case
            dtrain=train, # Specify training dataset
            num_boost_round=1000, # Set maximum iterations
            evals=[(valid, "validation")], # Specify evaluation dataset
            early_stopping_rounds=50 # If 50 iterations go without improvement early stop is done
        )

        y_pred=booster.predict(valid) # Once model trained, make predictions on validation set
        rmse=mean_squared_error(y_val, y_pred, squared=False) # RMSE to determine performance
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}
    
search_space = {
    # Controls depth of trees (converts float to int inside)
    'max_depth': scope.int(hp.quniform('max_depth',4,100,1)),
    # Learning rate from exp(-3), exp(0) basically [0.05 to 1]
    'learning_rate': hp.loguniform('learning_rate',-3,0),
    # Using similar loguniform logic as above for alpha and weight
    'reg_alpha': hp.loguniform('reg_alpha',-5,-1),
    'reg_lambda': hp.loguniform('reg_lambda',-6,-1),
    'min_child_weight': hp.loguniform('min_child_weight',-1,3),
    'objective': 'reg:linear',
    'seed':42,
}

best_result = fmin(
    # Pass all of the above to the fmin function which will minimize output to find best method
    fn=objective, # Define the function we created above
    space=search_space, # Pass search space
    algo=tpe.suggest, # Optimization algorithm we are running
    max_evals=50,
    trials=Trials() # Where information of each run is stored
)   
```

Now these experiments can be tracked in MLflow.

If you select all runs then click compare you will be taken to an experiment tracking comparison page like below.

Notice I have chosen 3 of the parameters to evaluate against the rmse metric.

You can see circled I clicked a portion of the rmse to filter by which clearly shows that there is a correlation with low min_child_weight corresponding to low rmse but not so much for max_depth. This is a powerful exploratory tool. 

![w2_image1](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/images/w2_image1.png)

Let's take a look at both of these parameters vs the rmse in a scatter plot. This will show you a clear correlation with min_child_weight as opposed to no clear correlation with max_depth.

![w2_image2](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/images/w2_image2.png)

![w2_image3](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/images/w2_image3.png)

Since we are using one of the approved models listed in the docs https://mlflow.org/docs/latest/tracking/autolog.html we can use auto logging.

As an example let's take the parameters from the best run (lowest rmse) in the GUI.

```python
params = {
    'learning_rate': 0.09596878839626283,
    'max_depth': 58,
    'min_child_weight': 1.7851046492749176,
    'objective': 'reg:linear',
    'reg_alpha': 0.013442382444691226,
    'reg_lambda': 0.06861630187449783,
    'seed': 42
}
```

Instead of this approach:
```python
with mlflow.start_run():
    mlflow.set_tag("model", "xgboost")
    mlflow.log_params(params)
    booster = xgb.train(
        params=params, # Specify parameters which are input variables in this case
        dtrain=train, # Specify training dataset
        num_boost_round=1000, # Set maximum iterations
        evals=[(valid, "validation")], # Specify evaluation dataset
        early_stopping_rounds=50 # If 50 iterations go without improvement early stop is done
    )
```

Let us instead use auto logging:
```python
mlflow.xgboost.autolog()

booster = xgb.train(
    params=params, # Specify parameters which are input variables in this case
    dtrain=train, # Specify training dataset
    num_boost_round=1000, # Set maximum iterations
    evals=[(valid, "validation")], # Specify evaluation dataset
    early_stopping_rounds=50 # If 50 iterations go without improvement early stop is done
)
```

Notice in the MLflow GUI that there are now 12 parameters and 3 metrics as opposed to the few we manually selected previously.

### Model Management with MLflow

Something like saving models in folders is very error prone and has no versioning or model lineage.

Previously we had a simple model output where it was saved in a local folder:

```python
# Since our Linear Regression model performed okay let's export it
with open('../models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out) # Save vectorizer and model
```

Using MLflow we can log these artifacts and others can download them directly from MLflow UI:

```python
mlflow.xgboost.autolog(disable=True)

with mlflow.start_run():
    
    #using same params as previously
    params = {
        'learning_rate': 0.09596878839626283,
        'max_depth': 58,
        'min_child_weight': 1.7851046492749176,
        'objective': 'reg:linear',
        'reg_alpha': 0.013442382444691226,
        'reg_lambda': 0.06861630187449783,
        'seed': 42
    }
    
    mlflow.log_params(params)

    booster = xgb.train(
        params=params, # Specify parameters which are input variables in this case
        dtrain=train, # Specify training dataset
        num_boost_round=1000, # Set maximum iterations
        evals=[(valid, "validation")], # Specify evaluation dataset
        early_stopping_rounds=50 # If 50 iterations go without improvement early stop is done
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    # Pull out the pre-processing so that it is saved in MLflow model management
    with open("models/preprocessors.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="w2_models_mlflow")
```

Note most of the code above was copied from before. The important portions:
- `mlflow.xgboost.log_model(booster, artifact_path="w2_models_mlflow")`
    - this portion allows the logging of the artifacts making this reproducible.
- `mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")`
    - this portion logs the data preprocessing as an artifact

![w2_image4](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/images/w2_image4.png)

If you don't use anaconda you can still see the dependencies in `requirements.txt`.

### Making a prediction

We have covered adding experiments to MLflow but now it is important to cover using them to make predictions.

From the documentation:

Predict with Spark DataFrame:
```python
import mlflow
logged_model = 'runs:/{runid}/models_mlflow'

# Load model into spark UDF
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on Spark DataFrame
columns = list(df.columns)
df.withColumn('predictions', loaded_model(*columns)).collect()

```

Predict with Pandas DataFrame:
```python
import mlflow
logged_model = 'runs:/{runid}/models_mlflow'

# Load model as PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame
import pandas as pd
loaded_model.predict(pd.DataFrame(data))

```