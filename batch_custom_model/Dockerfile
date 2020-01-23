FROM rio05docker/tflite_rpi:rpi3_test_6

RUN mkdir $HOME/scripts/ml
RUN mkdir $HOME/scripts/ml/models
RUN mkdir $HOME/scripts/ml/data

COPY main.py $HOME/scripts/ml 
COPY data/data.py $HOME/scripts/ml/data
COPY model/model.py $HOME/scripts/ml/model
COPY train.py $HOME/scripts/ml

ENTRYPOINT [ "" ]
