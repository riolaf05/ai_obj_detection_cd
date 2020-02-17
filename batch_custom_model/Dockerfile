FROM rio05docker/tflite_rpi:rpi3_test_6

LABEL "architecture" : armv7_linux

#Install git
RUN apt-get update && apt-get install -y git curl

RUN mkdir /home/scripts/ml
RUN mkdir /home/scripts/ml/models
RUN mkdir /home/scripts/ml/data
RUN mkdir /home/scripts/ml/input

RUN pip3 install mlflow

