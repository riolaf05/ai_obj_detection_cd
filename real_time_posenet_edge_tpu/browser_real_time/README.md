### pose_detection_tpu

Object detection inference with Edge TPU

Based on Raspbian Stretch, it contains:

* Python 3.5.3
* Tensorflow Lite
* Picamera module
* Edge TPU runtime

It must run on RaspberryPi with Edge TPU Coral device

To build: 

```console
docker build -t rio05docker/pose_detection_tpu:v1 .
docker push rio05docker/pose_detection_tpu:v1
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
docker run -it -p 8000:8000 --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(HOME)/Codice/ai_obj_detection_cd/real_time_posenet_edge_tpu/models:/home/scripts/pose_detection/browser/models rio05docker/pose_detection_tpu:v1 python3 /home/scripts/pose_detection/browser/opencv_pose_recognition_v2.py
```

Then log in on: `http://<<rpi3_ip>>:8000`

### TODO: 
* ~~Test real time predictions~~
* Add CI/CD for batch edge TPU
* ~~Add transfer learning with Edge TPU API~~
* Add MLFlow logging and packaging
