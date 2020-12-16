FROM nvcr.io/nvidia/l4t-ml:r32.4.2-py3

RUN apt-get update 

RUN apt-get install -y python3-pip build-essential curl

RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -

RUN apt install nodejs -y

RUN pip3 install --upgrade "elyra<2.0.0" 

RUN jupyter lab build

