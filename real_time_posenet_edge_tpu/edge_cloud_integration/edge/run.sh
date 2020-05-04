VIDEO=$1
POSE=$2
BUCKET=$3

docker run -it -v $(pwd)/images:/images -v $(pwd)/video:/video rio05docker/pose_detection_tpu:preprocess python3 /src/video_preoprocess.py --video=$VIDEO
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq -v $(pwd)/images:/home/scripts/pose_detection/images rio05docker/pose_detection_tpu:batch python3 /home/scripts/pose_detection/main.py --pose $POSE --bucket $BUCKET
echo "DONE"
