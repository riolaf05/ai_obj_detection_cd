### To build

docker build -t rio05docker/obj_detection_cd:rpi3_custom_test .
docker push rio05docker/obj_detection_cd:rpi3_custom_test
docker run -it --rm -p 8153:8153 rio05docker/obj_detection_cd:rpi3_custom_test bash
