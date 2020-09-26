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
docker build -t rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo .
docker push rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images -v /Codice/ai_obj_detection_cd/real_time_posenet_edge_tpu/models:/home/scripts/pose_detection/models rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo python3 /home/scripts/pose_detection/simple_pose.py
```

To run on custom image:
```console
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images -v /Codice/ai_obj_detection_cd/real_time_posenet_edge_tpu/models:/home/scripts/pose_detection/models rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo python3 /home/scripts/pose_detection/simple_draw_point.py --image /home/scripts/pose_detection/images/demo.jpg
```

Then log in on: `http://<<rpi3_ip>>:8080`

### TODO: 
* ~~Test real time predictions~~
* Add CI/CD for batch edge TPU
* ~~Add transfer learning with Edge TPU API~~
* Add MLFlow logging and packaging
