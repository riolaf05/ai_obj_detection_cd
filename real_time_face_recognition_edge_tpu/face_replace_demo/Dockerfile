FROM raspbian/stretch

LABEL 'maintainer' : 'rio05docker'

RUN apt-get update 
RUN apt-get install -y python3-dev 
RUN apt-get install -y python3-pip
RUN apt-get install -y curl zip wget 

COPY mobilenet_ssd_v2 /mobilenet_ssd_v2
COPY face_replace /face_replace

RUN apt-get install apt-transport-https -y
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - 
RUN apt-get update
RUN apt-get install libedgetpu1-std -y
RUN apt-get install python3-edgetpu -y

RUN wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl \
&& pip3 install tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl \
&& rm tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl

RUN apt-get install libcblas-dev -y
RUN apt-get install libhdf5-dev -y
RUN apt-get install libhdf5-serial-dev -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libjasper-dev -y
RUN apt-get install libqtgui4 -y 
RUN apt-get install libqt4-test -y
RUN apt-get install libilmbase-dev -y
RUN apt-get install libopenexr-dev -y
RUN apt-get install libgstreamer1.0-dev -y

RUN pip3 install imutils
RUN pip3 install opencv-python
#RUN pip3 install scipy
RUN pip3 install picamera
RUN pip3 install numpy
RUN pip3 install Pillow
