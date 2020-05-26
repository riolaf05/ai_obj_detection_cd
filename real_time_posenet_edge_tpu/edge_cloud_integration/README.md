### Activity Recognition on Edge TPU 

This project aims to perform activity recognition inference from pose estimation on edge devices with TPU, using ML pipeline for training steps on GCP.

Workflow:

![image](https://github.com/riolaf05/ai_obj_detection_cd/blob/develop/real_time_posenet_edge_tpu/edge_cloud_integration/activity_recognition_edge_cloud.jpg)

To run edge device pipeline (RaspberryPi3 with Coral Edge TPU):

1. Put video on `edge/video` folder.

2. `edge/run.sh <video_name> <pose_name> <bucket_name>`, this step will produce video embeddings and put them on provided GCP bucket.

3. Run Kubeflow pipeline, run `cloud/src/activity_classification.ipynb`


### TODO: 
* ~~Orchestrate edge and cloud pipelines~~
* Add connection between body points during pose recognition step
* Manage presence of people during pose recognition step
* Evaluate the use of LSTM RNN for activity classification
* Improve classification performance
* Fix visualizations on Kubeflow pipeline
