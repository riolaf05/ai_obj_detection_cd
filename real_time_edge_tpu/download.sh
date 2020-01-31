#!/bin/bash

#Note:
# $# is the number of arguments in $*
# $* is the list of arguments passed to the current process

if [ $# -eq 0 ]; then #this checks if any argument has been passes to the script
  DATA_DIR="/tmp"
else
  DATA_DIR="$1"
fi

# Install required packages
#python3 -m pip install -r requirements.txt

# Get TF Lite model and labels
curl -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ${DATA_DIR}
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# Get a labels file with corrected indices, delete the other one
(cd ${DATA_DIR} && curl -O  https://dl.google.com/coral/canned_models/coco_labels.txt)
rm ${DATA_DIR}/labelmap.txt

# Get version compiled for Edge TPU
(cd ${DATA_DIR} && curl -O  https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite)

echo -e "Files downloaded to ${DATA_DIR}"


#Extra line added in the script to run all command line arguments
exec "$@";
