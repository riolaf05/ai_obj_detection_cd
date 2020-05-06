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
docker build -t rio05docker/pose_detection_tpu .
docker push rio05docker/pose_detection_tpu
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images rio05docker/pose_detection_tpu python3 /home/scripts/pose_detection/main.py
```

To save on GCP cloud:

```console
export CREDENTIALS_JSON=<credential.json>

docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images -v $(pwd)/${CREDENTIALS_JSON}:/home/scripts/pose_detection/${CREDENTIALS_JSON} -e GOOGLE_APPLICATION_CREDENTIALS=home/scripts/pose_detection/${CREDENTIALS_JSON} rio05docker/pose_detection_tpu python3 /home/scripts/pose_detection/simple_pose.py
```


Then log in on: `http://<<rpi3_ip>>:8080`

### TODO: 
* ~~Test real time predictions~~
* Add CI/CD for batch edge TPU
* ~~Add transfer learning with Edge TPU API~~
* Add MLFlow logging and packaging
