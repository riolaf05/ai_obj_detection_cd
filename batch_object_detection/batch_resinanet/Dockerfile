FROM 

RUN pip3 install -r requirements.txt

# installing RetinaNet
RUN git clone https://github.com/fizyr/keras-retinanet.git
RUN cd keras-retinanet/
RUN pip install .

RUN pip install --upgrade keras

RUN python setup.py build_ext --inplace

RUN gdown --id 1mTtB8GTWs74Yeqm0KMExGJZh1eDbzUlT --output indian_number_plates.json

RUN cd ..
RUN mkdir snapshots

RUN gdown --id 1wPgOBoSks6bTIs9RzNvZf6HWROkciS8R --output snapshots/resnet50_csv_10.h5

RUN mkdir number_plates



