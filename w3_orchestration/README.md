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

### Ingestion 

##

##