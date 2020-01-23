FROM rio05docker/tflite_rpi:rpi3_test_6

RUN mkdri $HOME/scripts/ml

COPY main.py $HOME/scripts/ml 
COPY data.py $HOME/scripts/ml
COPY model.py $HOME/scripts/ml
COPY train.py $HOME/scripts/ml

