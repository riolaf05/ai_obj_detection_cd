FROM 

LABEL

RUN apt-get install -y -qq protobuf-compiler python-pil python-lxml python-tk

RUN pip3 -r requirements.txt

RUN mkdir /object_detection/
RUN mkdir /object_detection/data/
RUN mkdir /object_detection/models/
RUN mkdir /object_detection/data/label/
RUN mkdir /object_detection/data/label/train
RUN mkdir /object_detection/data/label/test
RUN mkdir /object_detection/data/annotations
RUN mkdir /object_detection/data/resized_images/
RUN mkdir /object_detection/data/resized_images/train
RUN mkdir /object_detection/data/resized_images/test

COPY src/ /object_detection/


