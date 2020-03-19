FROM tensorflow/tensorflow:1.12.0-rc2-devel

LABEL "architecture" : "x86"
LABEL "mainteiner" : "riolaf05"

# Install object detection api dependencies
RUN apt-get update && \
    apt-get install -y python python-tk wget emacs nano
RUN pip install Cython && \
    pip install contextlib2 && \
    pip install pillow && \
    pip install lxml && \
    #pip install jupyter && \
    pip install matplotlib

# Get protoc 3.0.0, rather than the old version already in the container
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

#RUN apt-get install -y -qq protobuf-compiler python-pil python-lxml python-tk git

#RUN pip3 -r requirements.txt

RUN mkdir /object_detection/

WORKDIR /object_detection/

RUN git clone --quiet https://github.com/tensorflow/models.git


