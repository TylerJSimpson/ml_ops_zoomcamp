# Introduction

## Environment Preparation

Please note I am using Linux OS (Ubuntu) via WSL on my local machine. 

Intel architecture is recommended over arm.

### Python

We are using Anaconda but any Python distribution should work.

Download:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
```

Execute installation:
```bash
bash Anaconda3-2024.02-1-Linux-x86_64.sh
```

Initialize the installation when download completes.

Test Anaconda is working:
```bash
which python
```

### Docker

Docker install
```bash
sudo apt install docker.io
```

We will be adding Docker compose to a local folder in our project:
```bash
mkdir software
cd software
```

Docker Compose download:
```bash
wget https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -O docker-compose
```

Make Docker Compose executable
```bash
chmod +x docker-compose
```

Allow access to 'software' folder from any location:
```bash
nano .bashrc
```

Add the following text to the '.bashrc':
```txt
export PATH="${HOME}/ml_ops_zoomcamp/software/"
```

After you are completed execute the '.bashrc':
```bash
source. bashrc
```

Test that Docker is working:
```bash
sudo docker run hello-world
```

You can add your use to Docker group to not have to use sudo each time:
```bash
sudo groupadd docker
```
```bash
sudo usermod -aG docker $USER
```
Note you will need to log out and back in for these previous 2 changes to work.

Test that Docker is working without sudo:
```bash
docker run hello-world
```

## Preparing Model

### Getting Data

All data is listed here:
https://github.com/DataTalksClub/nyc-tlc-data/releases/tag/green

Get January and February 2021 data from the above link:
```bash
wget https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_2021-01.csv.gz
wget https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_2021-02.csv.gz
```

Unzip the .gz:
```bash
gunzip green_tripdata_2021-01.csv.gz
gunzip green_tripdata_2021-02.csv.gz
```

### EDA and Model Development

This is done in Jupyter Notebook:
```bash
jupyter notebook
```

Follow along in the corresponding [notebook](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/notebooks/w1_duration-prediction.ipynb)

The model developed in the [notebook](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/notebooks/w1_duration-prediction.ipynb) was exported to a [model](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/models/lin_reg.bin)

## Model Overview

There are a few issues with our basic notebook. Note that we are using a basic model because the purpose of this course is ML ops not necessarily the model itself. 

1. We tested both Lasso and Ridge Linear Regression with multiple alpha values. This history is lost by the nature of the notebook. We could have kept track of the models but we did not. Logging the lr outputs with an experiment tracker would have been optimal.

2. When using our `with open() as f_out` to save the model we should have used a model registry to track and save models such as MLFlow. We will review this in [week 2](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w2_experiment_tracking).

3. We already put many cells into a function but we should have parameterized the input data so we could pick different dates easily. We review the rules and best practices of these ML Pipelines in [week 3](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w3_orchestration).

    Example workflow/ML pipeline:

    1. Load & prepare data
    2. Vectorize
    3. Train

4. Further, we need to consider deployment of the model. We review this in [week 4](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w4_deployment).

5. Once the model is deployed and in use we need to continue to monitor the model. This will be covered in [week 5](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w5_monitoring).

## Maturity Model

The highest level of ML Ops maturity involves all of the steps from the previous Model Overview section and automates each.  

Microsoft has created an [ML Ops maturity model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model) which has 5 levels of automation.


### Level 0: no ML Ops
- no automation
- all code in jupyter notebooks

Usually a solo data scientist without an engineering team. 

This is fine for proof of concept level but not production.

### Level 1: DevOps, no ML Ops
- releases are automated
- CI/CD
- unit & integration tests
- Ops metrics

Likely experienced developers using engineering/devops practices supporting a data scientist but they are operating separately in silos.

Usually can deploy a model similar to how you would other code, i.e. automated releases, CICD, unit testing. 

No experiment tracking and ML specific operations.

This maturity level can move from POC to production and work okay.

### Level 2: automated training
- training pipeline
- experiment tracking
- model registry
- low friction deployment

Usually data scientist and engineers are working together and a part of the same team.

Investing in this sort of infrastructure makes sense when you have multiple models in production. It may be too early to get to this maturity level if you are handling only 1-2 models in production.

### Level 3: automated deployment
- easy to deploy model
- A/B tests
- model monitoring

This makes sense when you have 2-3 mature use cases or perhaps 5+ standard use cases.

### Level 4: full ML Ops automation

Can automatically see models are drifting, trigger training, and deploying of the new model.

## Homework

Get January and February 2023 yellow taxi data for the homework:
```bash
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet
```

Unzip the .gz:
```bash
gunzip green_tripdata_2021-01.csv.gz
gunzip green_tripdata_2021-02.csv.gz
```

Week 1 homework can be found [here](https://github.com/TylerJSimpson/ml_ops_zoomcamp/tree/master/w1_into/w1_homework.ipynb)