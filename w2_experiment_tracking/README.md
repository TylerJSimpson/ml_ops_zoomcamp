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

The new notebook can be found [here](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w2_experiment_tracking/w1_duration-prediction.ipynb).

### Experiment Tracking with MLflow

We will be adding an additional experiment to the [notebook](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w2_experiment_tracking/w1_duration-prediction.ipynb). Key portions of the notebook will be listed below as they are added to the notebook.





