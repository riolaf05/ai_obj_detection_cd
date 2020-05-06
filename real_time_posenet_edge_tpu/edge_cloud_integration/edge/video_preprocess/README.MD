### rpi3_rt_tflite_tpu

Video preprocess with OpenCV

It must run on RaspberryPi 

To build: 

```console
docker build -t rio05docker/pose_detection_tpu:preprocess .
docker push rio05docker/pose_detection_tpu:preprocess
```

```console
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/images -v $(pwd)/video:video rio05docker/pose_detection_tpu:preprocess python3 /src/video_preoprocess.py --video <name.mp4> --bucket <bucket>
```
