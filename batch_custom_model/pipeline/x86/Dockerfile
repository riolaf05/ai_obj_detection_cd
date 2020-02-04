FROM ubuntu:18.04

LABEL "architcture" : x86_64_linux

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install keras
RUN pip3 install tensorflow
RUN apt-get install -y git curl

#Install GoCD
RUN echo "deb https://download.gocd.org /" | tee /etc/apt/sources.list.d/gocd.list
RUN curl https://download.gocd.org/GOCD-GPG-KEY.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y go-server

RUN apt-get install -y wget

#Install DVC
RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
RUN apt-get update
RUN apt-get install -y dvc

#Install MLFlow
RUN pip3 install mlflow

RUN mkdir /home/scripts
RUN mkdir /home/scripts/ml
RUN mkdir /home/scripts/ml/models
RUN mkdir /home/scripts/ml/data



