FROM ubuntu:18.04

COPY requirements.txt /requirements.txt
RUN apt-get update && apt-get instal -y python3-pip
RUN pip3 install -r requirements.txt

