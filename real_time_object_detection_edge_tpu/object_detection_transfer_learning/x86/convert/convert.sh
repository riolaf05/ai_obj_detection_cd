IMAGE_SIZE=300
FROZEN_GRAPH=$1
get_layers.py --frozen-graph=${FROZEN_GRAPH}
tflite_convert --output_file=tflite_graph.tflite --graph_def_file=${FROZEN_GRAPH} --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 --input_arrays=${FIRST_LAYER} --output_arrays=${LAST_LAYER} --allow_custom_ops