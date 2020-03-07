### Pose Detection of Edge TPU 

Object detection inference with Edge TPU

Based on Raspbian Stretch, it contains:

* Python 3.5.3
* Tensorflow Lite
* OpenCV
* Edge TPU runtime

It must run on RaspberryPi with Edge TPU Coral device

To build: 

```console
docker build -t rio05docker/obj_detection_cd:rpi3_posenet_browser_test .
docker push rio05docker/obj_detection_cd:rpi3_posenet_browser_test
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
docker run -it --privileged -p 8080:8080 -v /Codice/ai_obj_detection_cd/real_time_posenet_edge_tpu/models/:/home/scripts/pose_detection/browser/models -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -p 8080:8080 rio05docker/obj_detection_cd:rpi3_posenet_browser_test
```

Then log in on: `http://<<rpi3_ip>>:8080`

### TODO: 
* ~~Test real time predictions~~
* Add CI/CD for batch edge TPU
* ~~Add transfer learning with Edge TPU API~~
* (maybe?) Add MLFlow logging and packaging
