FROM ubuntu:18.04

LABEL "architecture" : x86_linux
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install keras
RUN pip3 install tensorflow
RUN apt-get install -y git curl

#Install git
RUN apt-get update && apt-get install -y git curl

RUN mkdir /home/scripts
RUN mkdir /home/scripts/ml
RUN mkdir /home/scripts/code
RUN mkdir /home/scripts/ml/models
RUN mkdir /home/scripts/ml/output
RUN mkdir /home/scripts/ml/input

RUN pip3 install mlflow

#Install GoCD server
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

#Install GoCD agent
RUN  echo "deb https://download.gocd.org /" | tee /etc/apt/sources.list.d/gocd.list
RUN curl https://download.gocd.org/GOCD-GPG-KEY.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y go-agent

WORKDIR /home/scripts/code

ENTRYPOINT nohup mlflow ui & && service go-server start && service go-agent start



