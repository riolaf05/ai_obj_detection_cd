FROM rio05docker/tflite_rpi:pose_detection_tpu_test

RUN mkdir /home/scripts/pose_detection/browser/
COPY browser__real_time/ /home/scripts/pose_detection/browser/
WORKDIR /home/scripts/pose_detection/browser/

RUN pip3 install -r requirements.txt
RUN chmod +x download_model.sh
RUN download_model.sh

CMD ["python3", "opencv_pose_recognition.py"]
