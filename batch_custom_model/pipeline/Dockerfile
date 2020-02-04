FROM ubuntu:18.04

LABEL "architcture" : x86_64_linux

RUN apt-get update
RUN apt-get install python3.6
RUN pip3 install keras
RUN apt-get install -y git curl

#Install GoCD
RUN echo "deb https://download.gocd.org /" | sudo tee /etc/apt/sources.list.d/gocd.list
RUN curl https://download.gocd.org/GOCD-GPG-KEY.asc | sudo apt-key add -
RUN apt-get update
RUN apt-get install -y go-server

#Install DVC
RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
RUN apt-get update
RUN apt-get install -y dvc

