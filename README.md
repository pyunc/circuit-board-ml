# POC - circuit board classifier ml-project

==============================

## General Objetives

* This mini classifier project aims to build a machine learning API for a circuit board quality checking device using computer vision. 
* The stacked used for this project are Tensorflow, TensorflowLite, Flask, Docker, Streamlit and Kubernets. 


![circuit-board-ml](https://github.com/pyunc/circuit-board-ml/blob/main/documentation/circuit-board-ml.png)

## Specific Objetives

* Wrap the mentioned ml product using Docker and create an MODEL API using Flask/FastAPI
* 3 different client interface able to interact with:
    * Streamlit/HTML web interface for single predictions jobs
    * An endpoint able to batch prediction jobs 
    * An edge device such as Raspberry Pi for real time classification/detection 

![circuit-board-ml](https://github.com/pyunc/circuit-board-ml/blob/main/documentation/circuit-board-ml.png)

ml-model inspired by those nice github:

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
`uvicorn app:app --reload`

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