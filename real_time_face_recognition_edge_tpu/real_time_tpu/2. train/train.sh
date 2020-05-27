#prerequisites
git submodule init && git submodule update
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/models/research/slim
#Train with L2-Normalized embedding and L2-Normalized weights of fully connected layer using the default parameters on the ImageNet dataset.
DATASET_DIR=/tmp/imagenet 
CHECKPOINT_DIR=/tmp/train_logs 
FINETUNE_CHECKPOINT_PATH=/tmp/my_checkpoints/mobilenet_v1_quant.ckpt 

python3 classification/mobilenet_v1_l2norm_train.py \
    --quantize=True \
    --fine_tune_checkpoint=$FINETUNE_CHECKPOINT_PATH \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --dataset_dir=$DATASET_DIR \
    --freeze_base_model=True

#Evaluating it on the imagenet dataset.
CHECKPOINT_FILE=$CHECKPOINT_DIR/mobilenet_v1_l2norm.ckpt
python3 classification/mobilenet_v1_l2norm_eval.py \
    --quantize=True \
    --checkpoint_dir=$CHECKPOINT_FILE \
    --dataset_dir=$DATASET_DIR

#Saves out a GraphDef containing the architecture of the model and freeze the graph.
python3 classification/export_inference_graph_l2norm.py \
  --quantize=True \
  --output_file=/tmp/mobilenet_v1_l2norm_inf_graph.pb
cd tensorflow
bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/mobilenet_v1_l2norm_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet_v1_l2norm.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_mobilenet_v1_l2norm.pb \
  --output_node_names=MobilenetV1/Predictions/Reshape_1

#For the model with an L2Norm operator in the convolutional kernels, the frozen graph cannot be transformed to tflite model successfully. 
#Need to transform the frozen graph to get rid of this operator.
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/tmp/frozen_mobilenet_v1_l2norm.pb  \
  --out_graph=/tmp/frozen_mobilenet_v1_l2norm_optimized.pb  \
  --inputs=input --outputs=MobilenetV1/Predictions/Reshape_1 \
  --transforms='strip_unused_nodes
      fold_constants'