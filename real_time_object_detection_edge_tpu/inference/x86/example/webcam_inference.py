import cv2
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import mlflow 
import argparse
import os 
import time

bg_mode=False
dt_mode=False
rec=False
codec=cv2.VideoWriter_fourcc(*'MJPG') #inizializzazione codec
out=None

SAVE_PATH='example/saved_model/1602777973'

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

#eper catturare immagini da webcam serve un'istanza di VideoCapture()
cap = cv2.VideoCapture(0) #gli 0 nel caso in cui ci siano più webcam

#per ottenere l'immagine della webcam (singolo frame)
#ritorna il valore di ritorno (true se la cattura è stata eseguita con successo) ed il frame
ret, frame = cap.read()

if(ret):
    print("Connesso alla webcam")
else:
    print("Webcam non disponibile o ce n'è più di una")
    exit(0)

#stampo una foto
'''
cv2.imshow("webcam", frame)
#cv2.imwrite("webcam.jpg", frame)
cv2.waitKey(0)
'''

#load model, see: https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], SAVE_PATH)
    graph = tf.get_default_graph()

    input_tensor = graph.get_tensor_by_name("serving_default_keras_layer_input:0")
    output_tensor = graph.get_tensor_by_name("StatefulPartitionedCall:0")

    #Stream di immagini da webcam
    while(cap.isOpened()):
        _ , frame = cap.read()

        cv2.imshow('webcam', frame)
        k = cv2.waitKey(1) #col parametro 1 waitKey è usato in modo non bloccante, k contiene il valore in ACII premuto da tastiera
        
        dim=tuple([224, 224])

        #resize image
        frame=cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = np.array(frame).reshape(-1, 224, 224, 3)

        #inference
        labels=['Kitten', 'Rufy']
        feed_dict ={input_tensor:frame}
        prediction=sess.run(output_tensor,feed_dict)[0]
        index=np.argmax(prediction, axis=0)
        if prediction[index] > 0.7:
            print(labels[index])
        else:
            print('Nothing detected!')
        
        if(k==ord("q")): #ord consente di ottenere il valore in ASCII di un carattere
            break

if(out!=None):
    out.release() #per rilasciare il file di registrazione

cap.release()
cv2.destroyAllWindows()