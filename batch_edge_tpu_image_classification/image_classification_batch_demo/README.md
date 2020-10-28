### rpi3_tflite_tpu_test
Based on Raspbian Stretch, it contains:

* Python 3.5.3
* Tensorflow Lite
* Picamera module
* Edge TPU runtime


To build: 

```console
docker build -t rio05docker/obj_detection_cd:rpi3_tflite_tpu_test .
docker push rio05docker/obj_detection_cd:rpi3_tflite_tpu_test
```

To run with Raspberry Camera and Edge TPU usb device:

```console
docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb --device=/dev/vchiq --rm rio05docker/obj_detection_cd:rpi3_tflite_tpu_test python3.5 detect_picamera.py bash
```

To run on x86 (with Qemu):
```console
docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb -v /usr/bin/qemu-arm-static:/usr/bin/qemu-arm-static --device=/dev/vchiq --rm rio05docker/obj_detection_cd:rpi3_tflite_tpu_test python3.5 detect_picamera.py bash
```

The `demo_batch_image_classification.sh` script can be used to download models, sample images and labels. To use image classification demo:

```python
python3 classify_image.py --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels models/inat_bird_labels.txt --input images/parrot.jpg
```

### TODO: 
* Add CI/CD for batch edge TPU

### References: 
* https://coral.ai/docs/accelerator/get-started/
* https://coral.ai/docs/edgetpu/tflite-python/