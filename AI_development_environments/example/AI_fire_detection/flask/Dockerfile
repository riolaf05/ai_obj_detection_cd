FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-pip git
COPY requirements.txt /src/requirements.txt
RUN pip3 install -r /src/requirements.txt
RUN apt-get install -y python3-dev build-essential libssl-dev libffi-dev python3-setuptools 

EXPOSE 5000

RUN mkdir -p /model
RUN mkdir -p /uploaded_photo
RUN mkdir -p /api
COPY api/* /api/





