### rpi3_rt_tflite_tpu

Object detection inference with Edge TPU

Based on Raspbian Stretch, it contains:

* Python 3.5.3
* Tensorflow Lite
* Picamera module
* Edge TPU runtime

It must run on RaspberryPi with Edge TPU Coral device

To build: 

```console
docker build -t rio05docker/tflite_rpi:pose_detection_tpu_test .
docker push rio05docker/tflite_rpi:pose_detection_tpu_test
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
docker run -it --privileged -p 8000:8000 -v /Codice/ai_obj_detection_cd/real_time_posenet_edge_tpu/project-posenet/models/:/home/scripts/models -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -p 8080:8080 rio05docker/tflite_rpi:pose_detection_tpu_test python3 pose_detection/simple_pose.py
```

Then log in on: `http://<<rpi3_ip>>:8080`

### TODO: 
* ~~Test real time predictions~~
* Add CI/CD for batch edge TPU
* ~~Add transfer learning with Edge TPU API~~
* (maybe?) Add MLFlow logging and packaging
