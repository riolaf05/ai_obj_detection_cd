# ML - DevOps CI/CD and data pipelines for Machine Learning inferences on RaspberryPi

This repo includes many code and data pipelines which can be used with MLFlow and Kubeflow frameworks to train and deploy several ML models for edge AI applications. 

### Instructions

This repo contains continuos integration pipelines (based on GitHub Actions) which are used to build Tensorflow, Keras and MLFlow based docker containers for ARM architectures. 

This is the general workflow which is followed by most of the covered use cases:
 
1. Put your data in the specified folders.
2. Run CICD pipeline to update Docker containers.
3. Run pre-processing, training and post-processing with **MLFlow** or **Kubeflow** provided data pipelines.
4. Get your model or result data from output folders.

Provided pipelines:

* Custom algorithm training using *MLFlow Project*
* Batch image recognition retrain ~~with weight imprinting~~ and transfer learning with *MLFlow Project* and inference on *Coral edge TPU*
* Real time object detection retrain using *Kubeflow* pipelines and inference on *Coral edge TPU*
* Real time pose estimation demo on Coral edge TPU and activity recognition training using *Kubeflow* pipelines 
* Real time face recognition demo on Coral edge TPU

### Agenda

* ~~fix pipeline errors~~
* ~~add model evaluation using MLFlow into pipeline flow: use another docker for evaluation and find a way to offer MLFlow UI~~
* ~~add CICD pipelines~~
* ~~test object detection and image recognition models [here](https://github.com/riolaf05/ai_obj_detection_cd/tree/adding-edge-tpu-/batch_masked_rcnn)~~
* fix tflite conversion for real time object detection from re-trained model
* fix Kubeflow pipeline errors for activity recognition 
* face recognition re-train

### References

* [Object detection transfer learning](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
* [Manage ML with MLFlow](https://thenewstack.io/tutorial-manage-machine-learning-lifecycle-with-databricks-mlflow/)
* [CD4ML](https://martinfowler.com/articles/cd4ml.html)
