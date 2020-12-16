import cv2
from datetime import datetime
import argparse
import os 
import time

bg_mode=False
dt_mode=False
rec=False
codec=cv2.VideoWriter_fourcc(*'MJPG') #inizializzazione codec
out=None

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

#Stream di immagini da webcam
while(cap.isOpened()):
    _ , frame = cap.read()

    cv2.imshow('webcam', frame)
    k = cv2.waitKey(1) #col parametro 1 waitKey è usato in modo non bloccante, k contiene il valore in ACII premuto da tastiera
    
    if(k==ord("q")): #ord consente di ottenere il valore in ASCII di un carattere
        break

if(out!=None):
    out.release() #per rilasciare il file di registrazione

cap.release()
cv2.destroyAllWindows()