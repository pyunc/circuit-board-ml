POC - circuit board classifier ml project

==============================

## Objetives

This mini classifier project aims to build a machine learning API using Flask, Docker, Streamlit and Kubernets. 

# batch and real time
# FastAPI
# Docker
# RestAPI and Streamlit
# TensorflowEdge 

![classification-examples-dog](https://github.com/pyunc/circuit-board-ml/blob/main/documentation/circuit-board-ml.png)


Project inspired by those nice github:

- https://towardsdatascience.com/image-classification-of-pcbs-and-its-web-application-flask-c2b26039924a
- https://github.com/utk-ink/Defect-Detection-of-PCB

## Initial setup or MakeFile

python -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

**or** 

`make install`

## Docker 

* build image

`docker image build --build-arg VERSION=0.0.1 -t circuit-board-ml .`

* spin the container

`docker run -d -p 5000:5000 -d circuit-board-ml`

* open in a browser 

`http://localhost:5000/`

## Python
Using Python 3.8.1

## Project organization 

    ├── Makefile           <- Makefile with commands like `make install`
    ├── requirements       
    ├── README.md   
    ├── Dockerfile             
    ├── main.yaml                               <- yaml file for AWS CodeBuild service test
    ├── notebook                                 <- Folder for Jupyter notebook for ML service and model dump
    │   ├── circuit-board-ml.ipynb    <- notebook for model train and dump
    │   ├── model.h5                        <- dumped model for pet (cat or dog classification)
    ├── webapp                                   <- Folder for the API ML 
    │   ├── app.py                               <- service in flask
    │   ├── model.h3                             <- dumped model
    │   ├── uploads                              <- folder responsible for receiving uploaded images from the template
    │   ├── templates                            <- html home page