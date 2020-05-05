#!/bin/bash

FROM 

LABEL 

RUN 

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install libedgetpu1-std -y

RUN wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
RUN rm tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl

RUN apt-get install libcblas-dev -y
RUN apt-get install libhdf5-dev -y
RUN apt-get install libhdf5-serial-dev -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libjasper-dev -y
RUN apt-get install libqtgui4 -y 
RUN apt-get install libqt4-test -y

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | RUN tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | RUN apt-key add -
RUN apt-get update
RUN apt-get install python3-edgetpu -y

RUN pip3 install imutils
RUN pip3 install opencv-python
RUN pip3 install scipy