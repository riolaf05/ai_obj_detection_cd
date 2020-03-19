docker build -t rio05docker/obj_detection_cd:x86_retrain_tflite .
docker push rio05docker/obj_detection_cd:x86_retrain_tflite
docker run -it -v ~/Codice/ai_obj_detection_cd/real_time_object_detection_edge_tpu&/object_detection_transfer_learning/x86/data:/object_detection/ --rm rio05docker/obj_detection_cd:x86_retrain_tflite python3 config_pipeline.py


