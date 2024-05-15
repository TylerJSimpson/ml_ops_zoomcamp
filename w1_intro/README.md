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