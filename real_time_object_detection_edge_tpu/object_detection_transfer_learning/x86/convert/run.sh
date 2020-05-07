BASE_DIR=~/Codice/ai_obj_detection_cd/real_time_object_detection_edge_tpu/object_detection_transfer_learning/x86
cd $BASE_DIR/convert
git clone https://github.com/tensorflow/models
./obj_det_scripts/convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 267

