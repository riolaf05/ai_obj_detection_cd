FROM 

LABEL

RUN apt-get install -y -qq protobuf-compiler python-pil python-lxml python-tk git

RUN pip3 -r requirements.txt

RUN mkdir /object_detection/

WORKDIR /object_detection/

RUN git clone --quiet https://github.com/tensorflow/models.git


