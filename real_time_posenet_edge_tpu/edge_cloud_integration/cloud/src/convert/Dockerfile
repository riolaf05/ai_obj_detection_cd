FROM tensorflow/tensorflow:latest-gpu-py3

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

RUN mkdir /src
COPY convert.py /src
COPY auth/ai-vqc-fb43ef70f5ec.json /ai-vqc-fb43ef70f5ec.json

WORKDIR /src

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ENV PYTHONPATH='/src/:$PYTHONPATH'
ENV GOOGLE_APPLICATION_CREDENTIALS=/ai-vqc-fb43ef70f5ec.json
