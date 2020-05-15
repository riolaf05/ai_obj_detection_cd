CONFIG_FILE=$1
IMAGE_SIZE=$2
#Export frozen graph
python /tensorflow/models/research/object_detection/export_inference_graph.py --input_type image_tensor --input_shape 1,300,300,3 --pipeline_config_path=/object_detection/preprocess/$CONFIG_FILE --trained_checkpoint_prefix training/output/model.ckpt-5 --output_directory /object_detectiontraining/output/
#Get first and last layer of frozen graph
get_layers.py --frozen-graph=/object_detectiontraining/output/frozen_inference_graph.pb
#Convert to TFLite model
tflite_convert --output_file=tflite_graph.tflite --graph_def_file=/object_detectiontraining/output/frozen_inference_graph.pb --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 --input_arrays=image_tensor --output_arrays=detection_boxes,detection_scores,detection_classes,num_detections --allow_custom_ops