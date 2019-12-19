FROM rio05docker/tflite_rpi:rpi3_test_6

RUN apt-get install -y git  
RUN git clone https://github.com/experiencor/kangaroo.git

RUN mkdir /home/scripts/samples

 