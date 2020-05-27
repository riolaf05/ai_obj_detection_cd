### Training workflow:

The workflow (executed manually or with MLFlow):

1. Clone this repository

2. Get new images (both for train and for test) to retrain base model (e.g. with [image_downloader](https://github.com/Swaini/object_detection_retraining/blob/master/image_downloader.py))

3. **Label** new images (e.g. with `LabelImg`) and create **label map**, put images in `data/images` folder

This steps produces a list of **bounding boxes** for the image. You can use [Labelimg](https://github.com/tzutalin/labelImg) for annotations. Labelimg is a great tool for drawing bounding boxes on images, the bounding boxes get automatically converted into xml documents which you can then use for tensorflow.

To install **Labelimg**:

```console
pip3 install PyQt5 \
git clone https://github.com/tzutalin/labelImg \
&& cd labelimg \
&& sudo apt-get install -y pyqt5-dev-tools \
&& sudo pip3 install -r requirements/requirements-linux-python3.txt \
&& make qt5py3 \
#python3 labelImg.py
```
Create a **label map** which defines a mapping from string class names to integer class Ids. The label map should be a `StringIntLabelMap` text protobuf.

e.g.:

```json
item {
  id: 1
  name: 'Abyssinian'
}

item {
  id: 2
  name: 'american_bulldog'
}

item {
  id: 3
  name: 'american_pit_bull_terrier'
}
```

4. Launch run.sh which executes (it uses **MLFlow**):

* Launch preprocess component. Object Detection retrain config file is stored in:`ai_obj_detection_cd\real_time_object_detection_edge_tpu\object_detection_transfer_learning\x86\training`

* Launch re-train component. Checkpoint files are stored in: `ai_obj_detection_cd\real_time_object_detection_edge_tpu\object_detection_transfer_learning\x86\training\checkpoints`

* Lauch convert script which convert checkpoint files into a frozen graph, then convert that graph to a TensorFlow Lite flatbuffer file Tensorflow Lite model (`.tflite`).

The network_type can be either `mobilenet_v1_ssd`, or `mobilenet_v2_ssd`.

If the following problem occurs:

```console
Tensorflow Allocation Memory: Allocation of 38535168 exceeds 10% of system memory
```

Launch the `reset_graph.py` script inside the container.

Per visualizzare il grafo Tensorflow usare [Netron](https://github.com/lutzroeder/netron).

### Notes about saving Tensorflow Models:
From [Tensorflow Checkpoints](https://www.tensorflow.org/guide/checkpoint)

The phrase "Saving a TensorFlow model" typically means one of two things:

1. Checkpoints
2. SavedModel

Checkpoints capture the exact value of all parameters (`tf.Variable objects`) used by a model. Checkpoints do not contain any description of the computation defined by the model and thus are typically only useful when source code that will use the saved parameter values is available.

The SavedModel format on the other hand includes a serialized description of the computation defined by the model in addition to the parameter values (checkpoint). Models in this format are independent of the source code that created the model. They are thus suitable for deployment via **TensorFlow Serving**, **TensorFlow Lite**, **TensorFlow.js**, or programs in other programming languages (the C, C++, Java, Go, Rust, C# etc. TensorFlow APIs).

### Notes about Object Detection Retrain:
Object Detection Retrain API uses config files to build retrain pipeline. See [docs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

### Notes about MLFlow:
When you run an MLflow project that specifies a Docker image, MLflow adds a new Docker layer that copies the project’s contents into the /mlflow/projects/code directory. This step produces a new image. MLflow then runs the new image and invokes the project entrypoint in the resulting container.

### TODO
* ~~Define automatic retrieval of boxes data from xml or csv~~
* Convert SavedModel to Tensorflow Lite
* Add parameters to choose base model
* Add GPU support on training step

### References
* [Coral Edge TPU docs](https://coral.ai/docs/edgetpu/retrain-detection/#start-training)
* [Retrain Object Detection Models](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
* [Tensorflow: using your own dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
* [Using custom dataset with Tensorflow](https://aaronjencks.blogspot.com/2019/03/using-custom-datasets-with-tensorflows.html#TFR)
* [Tensorflow: TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord)
* [Working with TFRecords](https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d)
