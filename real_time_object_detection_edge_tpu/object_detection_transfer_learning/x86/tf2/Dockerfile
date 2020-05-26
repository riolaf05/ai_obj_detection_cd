FROM tensorflow/tensorflow:latest-gpu-py3

LABEL "architecture" : "x86"
LABEL "mainteiner" : "riolaf05"

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

RUN apt-get update && apt-get install python3-pip -y

RUN mkdir /src
RUN mkdir /data
RUN mkdir /save

COPY requirements.txt /src/requirements.txt

RUN pip3 install -r /src/requirements.txt

COPY preprocess/resize_images.py /src/resize_images.py
COPY train/train.py /src/train.py
COPY convert/convert_cpu.py /src/convert.py
COPY compile/compile.sh /src/compile.sh

WORKDIR /src

#Download Edge TPU compiler
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update && apt-get install edgetpu-compiler -y



