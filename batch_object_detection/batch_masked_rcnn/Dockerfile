FROM rio05docker/tflite_rpi:rpi3_test_6

RUN apt-get update && apt-get install -y git  
RUN git clone https://github.com/matterport/Mask_RCNN.git && cd Mask_RCNN && python3 setup.py install
RUN pip3 install scikit-image --user
RUN cd ..
RUN apt-get purge git -y
RUN rm -R Mask_RCNN

RUN mkdir /home/scripts/samples
COPY train/main.py /home/scripts/
COPY train/KangarooDataset.py /home/scripts/
COPY samples/ /home/scripts/samples

RUN pip3 install IPython

RUN wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -P /home/scripts/models

RUN apt-get autoremove -y 
RUN apt-get clean


 
