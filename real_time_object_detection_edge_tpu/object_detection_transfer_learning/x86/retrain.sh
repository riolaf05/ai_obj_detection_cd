git clone --quiet https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH="/object_detection/models/research/:/object_detection/models/research/slim:$PATH"
python3 /object_detection/builders/model_builder_test.py

python3 config_pipeline.py

python3 /object_detection/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}

