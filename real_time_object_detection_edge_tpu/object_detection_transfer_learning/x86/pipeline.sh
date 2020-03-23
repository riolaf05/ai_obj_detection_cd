mlflow run preprocess/ -b local --no-conda -e preprocess
mlflow run training/ -b local --no-conda -e train config_file='ssd_mobilenet_v2_coco.config' num_train_steps=500 num_eval_steps=500
