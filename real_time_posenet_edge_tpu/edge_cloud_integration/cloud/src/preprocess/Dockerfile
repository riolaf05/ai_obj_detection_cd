FROM python:3.6
LABEL 'maintainer' : 'rio'

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

COPY auth/ai-vqc-fb43ef70f5ec.json /ai-vqc-fb43ef70f5ec.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/ai-vqc-fb43ef70f5ec.json

RUN mkdir /src
RUN mkdir /src/jsons
COPY preprocess.py /src
WORKDIR /src

