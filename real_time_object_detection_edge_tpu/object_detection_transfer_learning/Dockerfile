FROM tensorflow/tensorflow:1.12.0-rc2-devel

LABEL "architecture" : "x86"
LABEL "mainteiner" : "riolaf05"

RUN apt-get install -y -qq protobuf-compiler python-pil python-lxml python-tk git

RUN pip3 -r requirements.txt

RUN mkdir /object_detection/

WORKDIR /object_detection/

RUN git clone --quiet https://github.com/tensorflow/models.git


