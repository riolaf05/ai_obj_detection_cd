FROM rio05docker/ai-toolkit:arm64_keras

RUN apt update
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config  libswscale-dev libtbb2 libtbb-dev

RUN pip3 install --upgrade pip
RUN pip3 install scikit-build
RUN pip3 install cmake
RUN pip3 install opencv-python
