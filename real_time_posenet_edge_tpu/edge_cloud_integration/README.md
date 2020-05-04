### Activity Recognition on Edge TPU 

Activity recognition inference with Edge TPU and model train on GCP

Workflow:

![image](https://github.com/riolaf05/\ai_obj_detection_cd/blob/develop/activity_recognition_edge_cloud.jpg)

```console
docker build -t rio05docker/obj_detection_cd:rpi3_posenet_browser_test .
docker push rio05docker/obj_detection_cd:rpi3_posenet_browser_test
```

To run demo with Raspberry Camera and Edge TPU usb device:

```console
git clone https://github.com/riolaf05/ai_obj_detection_cd
cd pose-detection/real_time_posenet_edge_tpu
docker run -it --privileged -p 8080:8080 -v models/:/home/scripts/pose_detection/browser/models -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -p 8080:8080 rio05docker/obj_detection_cd:rpi3_posenet_browser_test python3 opencv_pose_recognition.py
```

Then log in on: `http://<<rpi3_ip>>:8080`

### TODO: 
* ~~Test real time predictions~~
* ~~Add transfer learning with Edge TPU API~~
* Draw lines between body parts
* Add CI/CD for batch edge TPU
* (maybe?) Add MLFlow logging and packaging
