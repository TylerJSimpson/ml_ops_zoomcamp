# Orchestration

## Pre-requisites

1. Install requirements
```bash
pip install -U prefect
```

2. Start local Prefect server
```bash
prefect server start
```

Default URL:
`127.0.0.1:4200/dashboard`

## Introduction

### MLOps involves 4 steps

1. Preparing the model or deployment involves optimizing performance, ensuring it handles real-world data, and packaging it for integration into existing systems.

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

## Prefect Overview

### Introduction

Prefect allows you to orchestrate and observe your python workflows at scale.

It is a flexible open-source Python framework to turn standard pipelines into fault-tolerant dataflows.

When self hosting a Prefect server you have an `Orchestration API` that is a rest API that handles the metadata. The metadata is stored in a data base `sqlite` is the default.

### Terminology

* `tast` - a discrete unit of work in a Prefect workflow, consider this a regular python function
* `Flow` - serves as the container for workflow logic, consider this as the main python function that calls the others
* `Subflow` - `Flow` called by another `Flow`

## Prefect Operations

### File Setup

We took the original notebook `duration_prediction_original.ipynb` and added some exploration and saved it as `duration_prediction_explore.ipynb`.

We then work on converting this notebook to a python script in `orchestrate_pre_prefect.py`.

Finally we add our Prefect code which can be seen in the script `orchestrate.py`

All we need to do is add decorators notice the addition of either `@task` or `@flow` decorators. 

There are some other specifications you can call in the decorators including:
* retries (int)
* retry_delay_seconds (int)
* log_prints (bool)

### Projects

To initialize a project:
```bash
prefect project init
```

This will create:
* `.prefectignore`
* `deployment.yaml`: useful for templating/making multiple deployments
* `prefect.yaml`: 
* `.prefect/`: hidden folder for short hand convenience

Running `prefect project init` will not overwrite these files they will need to be deleted then re-created.

#### .yaml

`name` will be pulled from the folder name the project was initialized in.
`prefect-version` will come from the python version you are running.
`build` is used when working with docker images.
`push` is used when working with remote locations such as Azure/AWS.
`pull` shows the git repository and branch that the code will be pulled from.

#### Workers

Now we need to assign a worker.

This can be done in the UI via `Work Pools` then in this case `default agent`. We have named ours `zoompool`.

#### Starting a worker

prefect worker start is used for non prefect agent workers.

```bash
prefect agent start --pool 'zoompool'
```

#### Deploy flow

Specify file, main flow, name it and specify the worker pool:
```bash
prefect deploy 3.4/orchestrate.py:main_flow -n taxi1 -p zoompool
```
Note that if you specify something wrong like the worker pool or the pool is not already running, Prefect will prompt you through creating a work pool in the CLI.

#### Creating blocks

Blocks can be created in the UI or with .py files.

Note that depending on what system/API you are interacting with you will need to install the supporting packages.

Azure:
```bash
pip install prefect-azure
```

AWS:
```bash
pip install prefect-aws
```

In `3.5/create_s3_bucket_block.py` you can see example of creating blocks via code.

```python
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id="123abc", aws_secret_access_key="abc123"
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="my-first-bucket-abc", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
```