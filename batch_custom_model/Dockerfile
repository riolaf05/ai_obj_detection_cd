FROM rio05docker/tflite_rpi:rpi3_test_6

RUN mkdir /home/scripts/ml
RUN mkdir /home/scripts/ml/models
RUN mkdir /home/scripts/ml/data

COPY main.py /home/scripts/ml 
COPY data/data.py /home/scripts/ml/data
COPY model/model.py /home/scripts/ml/model
COPY train.py /home/scripts/ml

RUN pip3 install mlflow

ENTRYPOINT nohup mlflow ui & && python3 main.py
