### Activity Recognition on Edge TPU 

Activity recognition inference with Edge TPU and model train on GCP

Workflow:

![image](https://github.com/riolaf05/ai_obj_detection_cd/blob/develop/real_time_posenet_edge_tpu/edge_cloud_integration/activity_recognition_edge_cloud.jpg)

To run edge device pipeline (RaspberryPi3 with Coral Edge TPU):

1. Put video on `edge/video`

2. `edge/run.sh`

To run Kubeflow pipeline, run `cloud/src/activity_classification.ipynb`


### TODO: 
* ~~Orchestrate edge and cloud pipelines~~
* Add connection between body points
* Manage multiple pose recogniton
* Improve classification performance
* Fix visualizations on Kubeflow pipeline
