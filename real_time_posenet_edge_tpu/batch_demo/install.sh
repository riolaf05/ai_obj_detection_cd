docker build -t rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo .
docker push rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images rio05docker/ai_obj_detection_cd:pose_detection_tpu_demo python3 /home/scripts/pose_detection/simple_pose.py