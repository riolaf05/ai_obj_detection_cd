
tflite_convert \
  --output_file=/tmp/frozen_mobilenet_v1_l2norm_optimized.tflite \
  --graph_def_file=/tmp/frozen_mobilenet_v1_l2norm_optimized.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1