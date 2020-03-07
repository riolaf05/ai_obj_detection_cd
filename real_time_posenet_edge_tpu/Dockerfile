FROM rio05docker/tflite_rpi:pose_detection_tpu_test

RUN mkdir /root/scripts/pose_detection/browser/
COPY browser__real_time/ /root/scripts/pose_detection/browser/
WORKDIR /root/scripts/pose_detection/browser/

RUN pip3 install -r requirements.txt
RUN chmod +x download_model.sh
RUN download_model.sh

CMD ["python3", "opencv_pose_recognition.py"]