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

File `predict.py` contains the code for making the predictions.

```python
import pickle

from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

### Create script for testing the prediction script

File `test.py` contains the test input.

```python
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
```

### Testing 

Now, with the python environment activated, you can run the Flask app then send the test to it.

```bash
python predict.py
python test.py
```

And you should get the return value `{'duration': 26.2158369001914}`

Note you will be warned `WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.`

To get rid of this you can switch to gunicorn

Ensure gunicorn is installed

```python
pip install gunicorn
```

Instead of running `python predict.py` you can use gunicorn now to specify the same port and the file name.

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

Then you can once again run the test to this application:
```bash
python test.py
```
Returning result `{'duration': 26.2158369001914}`

### Containerizing in Docker

Dockerfile:

```
FROM python:3.9.19-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
```

Ensure that the python version matches what you have been developing in `python -V`.

There are versions other than slim that can be used but slim works well with Ubuntu.

Note that this line `COPY [ "Pipfile", "Pipfile.lock", "./" ]` means to copy the predict.py and lin_reg.bin to the directory above which is specified by `WORKDIR /app`.

Install the dependencies to the system `RUN pipenv install --system --deploy`.

Then you can copy the model `lin_reg.bin` and the `predict.py` script that calls it and handles the pre-processing and return value.

We then expose the port from the Flask app and set the entry point to gunicorn like we did in the CLI ad hoc before.


Use Dockerfile to build the image and specify the name:tag:
```bash
docker build -t ride-duration-prediction-service:v1 .
```

Run it in interactive mode `-it` and remove iamge when done `-rm` and map the port 9696 to the port 9696 on the host machine `9696:9696`:
```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

Now you can run the test manually to hit the docker container:
```bash
python test.py
```

### Combining deployment with model registry

Instead of passing the file like below, we should grab from the mlflow server:
```python
with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
```

1st ensure that you run mlflow server, note it depends what folder you run it from. We are using a local artifact store. It would be best in production to use object storage such as s3 or adls2.
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --port 5001
```

Now in `random-forest.ipynb` this pre-made model was grabbed and updated to use the correct port. The `dict_vectorizer.bin` was added to the artifacts after running and getting the run ID and manually placed. 

So now you will see in the mlflow UI that there is a run. You can promote this to production in the GUI via `register model` but we can also just use the run ID the workflow will be the same.

Now we are going to make a new directory and move our relevant files
```bash
mkdir webservice-mlflow
```
* `Pipfile`
* `Pipfile.lock`
* `predict.py`
* `test.py`
* `dict_vectorizer.bin`
* `random-forest.ipynb`
* `lin_reg.bin`

Now we can replace the local model code in the new `predict.py` below:
```python
with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
```

With this code:
```python
RUN_ID = '4ba9756ec7524ab9bff0d75232a15c9b'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5001'

client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
print('Downloading the dict vectorizer to {path}')

with open(path, 'rb') as f_out:
    dv = pickle.loan(f_out)

logged_model = f'runs:/{RUN_ID}/model'

model = mlflow.pyfunc.load_model(logged_model)
```

Now let's test. We do not have mlflow in the pipenv yet so let's move to the the new folder we created and make sure we install mlflow in the pipenv.

```bash
cd web-service-mlflow
pipenv install mlflow
```

Notice it picked up all of the necessary packages:
```
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
scikit-learn = "==1.0.2"
flask = "*"
gunicorn = "*"
mlflow = "*"

[dev-packages]
requests = "*"

[requires]
python_version = "3.9.19"
```

Now you can run from this location:
```bash
python predict.py
```

And you can test against it:
```bash
python test.py
```

Instead of having a dict vectorizer stored separately and a model it can be useful to combine them into a pipeline.