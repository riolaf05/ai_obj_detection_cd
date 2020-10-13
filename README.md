# RaspberryPi MLOps - DevOps CI/CD and data pipelines for Machine Learning on RaspberryPi

Differents pipelines architectures built on **MLFlow** and **Kubeflow** which can be used to train and deploy several ML models for **edge AI applications** (and x86). 

### Advantages of "MLOps"

1. best collaboration between data scientists and developers: the former does not need to worry about code.
2. model validation: easy and fast model parameters validation through UIs.
3. model reproducibility: model versioning and easy check with sets of data.
4. model deployment: automatic model depoyment with deploy pipelines.

### Instructions

The provided pipelines can be used to build Tensorflow, Keras and MLFlow based docker containers for ARM (and x86) architectures. 

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
* Custom NLP for chatbot 

### Agenda

* ~~fix pipeline errors~~
* ~~add model evaluation using MLFlow into pipeline flow: use another docker for evaluation and find a way to offer MLFlow UI~~
* ~~add CICD pipelines~~
* ~~test object detection and image recognition models [here](https://github.com/riolaf05/ai_obj_detection_cd/tree/adding-edge-tpu-/batch_masked_rcnn)~~
* fix Kubeflow pipeline errors for activity recognition 
* fix tflite conversion for real time object detection from re-trained model
* face recognition re-train
* develop new use cases
* build UI for pipeline selection and deploy

### References

* [Object detection transfer learning](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
* [Manage ML with MLFlow](https://thenewstack.io/tutorial-manage-machine-learning-lifecycle-with-databricks-mlflow/)
* [CD4ML](https://martinfowler.com/articles/cd4ml.html)
