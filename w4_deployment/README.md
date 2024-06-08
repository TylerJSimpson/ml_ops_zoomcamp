# Deployment

## Introduction

If real-time results are not necessary then a batch (offline) deployment will fit. Otherwise you will want an online deployment.

In online deployments there are some requirements
* web service
* stream service

### offline

Run the model regularly usually hourly, daily, monthly, etc.

Generally pulls data from a database and runs the data through a model and publishes the predictions. 

### online

#### Web service

1:1 client-server relationship. The client being the backend connects to the live service model 

#### Streaming

1:x producer to consumer relationship. You have a producer and multiple consumers. Producers push to an event stream and consumers react to the events.



## Deploying model as a web service

### Overview

Remember we created the pickle file `lin_reg.bin` previously.
This is the file we will use for deployment, I have copied it into the `w4_deployment` folder.

Steps:
* create virtual environment with `pipenv`
* create a script for predicting
* put script in `flask` app
* package the app with `docker`

### Create venv

Find versions of packages we used (just scikit learn for now)
```bash
pip freeze
```
The version: `scikit-learn==1.2.2`

Note we also need flask but the version is not important.

You can also specify the path to a python distribution instead of writing out the version.

```bash
pipenv install scikit-learn==1.2.2 flask --python=3.11.5
```

Activate the environment:
```bash
pipenv shell
```

Note that a `Pipfile` and `Pipfile.lock` were both created in your directory.

### Create script for predicting