# ML - Continuous Integration and Continuous Delivery pipelines for RaspberryPi
Continuous delivery pipeline for object detection models and apps

### Instructions

This is a multi-purpose ML continuos delivery pipeline based on GitHub Actions, Tensorflow, MLFlow. 

It can be used to train ML models then evaluate them using **MLFlow** or **TensorBoard**

This is the general pipeline which is followed by most of the covered use cases:

### TODO: review this steps!

1. Clone the repo into your local machine
2. Put your data in the specified folders
3. Push and start the continuous integration pipeline
4. Run the given docker container, watch the results with **MLFlow** or **TensorBoard** and get your model or results from output folders.

### Agenda

* ~~fix pipeline errors~~
* ~~add model evaluation using MLFlow into pipeline flow: use another docker for evaluation and find a way to offer MLFlow UI~~
* ~~add CICD pipelines~~
* test object detection and image recognition models [here](https://github.com/riolaf05/ai_obj_detection_cd/tree/adding-edge-tpu-/batch_masked_rcnn)
* test re-train with edge TPU real time object detection [here](https://github.com/riolaf05/ai_obj_detection_cd/tree/adding-edge-tpu-/real_time_edge_tpu/real_time_transfer_learning_tpu)
* ~~test real time object detection models [here](https://github.com/riolaf05/ai_obj_detection_cd/tree/adding-edge-tpu-/real_time_edge_tpu)~~
* smooth object detection docker image to be used on-demand

### References

* [Object detection transfer learning](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
* [Manage ML with MLFlow](https://thenewstack.io/tutorial-manage-machine-learning-lifecycle-with-databricks-mlflow/)
* [CD4ML](https://martinfowler.com/articles/cd4ml.html)
