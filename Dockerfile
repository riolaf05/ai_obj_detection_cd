FROM rio05docker/tflite_rpi:rpi3_test_6

RUN apt-get install -y git  
RUN git clone https://github.com/matterport/Mask_RCNN.git
RUN cd Mask_RCNN
RUN python3 setup.py install
RUN pip3 install scikit-image --user
RUN cd ..
RUN apt-get purge git -y
RUN rm -R Mask_RCNN

RUN mkdir /home/scripts/samples

 