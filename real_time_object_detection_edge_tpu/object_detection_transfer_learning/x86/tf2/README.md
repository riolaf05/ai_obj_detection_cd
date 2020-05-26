### Transfer Learning with Tensorflow 2.0 and MLFlow for Edge TPU inference
Object detection inference with Edge TPU

1. Create `preprocess`, `train`, `convert` code and run CICD pipeline. The actual code uses MobileNetV2 base model.

2. Put images on `data/` folder, put each image in a folder with the same name of the label.

3. Install MLFlow on your local machine:

```console
pip3 install mlflow
```

4. Run MLFlow for train (on GPU where possible):

```console
mlflow run . -b local --no-conda -e preprocess -P directory='<data directory>' -P size='800 600'
mlflow run . -b local --no-conda -e train -P image_size=224 -P batch_size=16 -P epochs=10
mlflow run . -b local --no-conda -e convert -P image_size=224
mlflow run . -b local --no-conda -e compile
```

5. Run inference on `image_recognition.ipynb` notebook with new images.

**Note**: you cannot train a model directly with TensorFlow Lite; instead you must convert your model from a TensorFlow file (such as a .pb file) to a TensorFlow Lite file (a `.tflite` file), using the TensorFlow Lite converter. The figure illustrates the basic process to create a model that's compatible with the Edge TPU. Most of the workflow uses standard TensorFlow tools. Once you have a TensorFlow Lite model, you then use our Edge TPU compiler to create a `.tflite` file that's compatible with the Edge TPU.

![TFLite Workflow](https://coral.ai/static/docs/images/edgetpu/compile-workflow.png)

See [model requirements](https://coral.ai/docs/edgetpu/models-intro/#model-requirements) for Edge TPU conversion.

Not all operations may be supported by the Edge TPU. A percentage of the model (e.g. `DEPTHWISE_CONV_2D` layer) could be runned on the CPU, which is slower.

### TODO: 
* ~~Test batch predictions~~
* Compile for Edge TPU and add real time inference script with OpenCV
* Add MLFlow logging and packaging
* Test run on Kubeflow
