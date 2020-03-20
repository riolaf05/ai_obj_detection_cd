### Commands

### Manual build/push
```console
cd ~/Codice/ai_obj_detection_cd/real_time_object_detection_edge_tpu/object_detection_transfer_learning/x86
git pull && docker build -t rio05docker/obj_detection_cd:x86_retrain_tflite .
docker push rio05docker/obj_detection_cd:x86_retrain_tflite
```

### PRETRAIN
```console
CODE_DIR=~/Codice/ai_obj_detection_cd/real_time_object_detection_edge_tpu/object_detection_transfer_learning/x86

docker run -it --rm -v $CODE_DIR/data:/object_detection/data -v $CODE_DIR/training:/object_detection/training rio05docker/obj_detection_cd:x86_retrain_tflite bash

python3 config_pipeline.py
```

### TRAIN
```console
NUM_TRAINING_STEPS=500 && NUM_EVAL_STEPS=100

python /tensorflow/models/research/object_detection/model_main.py \
  --pipeline_config_path="/object_detection/training/ssd_mobilenet_v2_coco.config" \
  --model_dir="/object_detection/training" \
  --num_train_steps="${NUM_TRAINING_STEPS}" \
  --num_eval_steps="${NUM_EVAL_STEPS}"
  ```
 