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