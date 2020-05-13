IMAGE_SIZE=$1
FROZEN_GRAPH=$2
#Export frozen graph
python /tensorflow/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path='/object_detection/preprocess/ssd_mobilenet_v2_coco.config' --trained_checkpoint_prefix training/output/model.ckpt-5.index --output_directory frozen_graph/
#Get first and last layer of frozen graph
get_layers.py --frozen-graph=${FROZEN_GRAPH}
#Convert to TFLite model
tflite_convert --output_file=tflite_graph.tflite --graph_def_file=${FROZEN_GRAPH} --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 --input_arrays=image_tensor --output_arrays=detection_boxes,detection_scores,detection_classes,num_detections --allow_custom_ops