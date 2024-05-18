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
