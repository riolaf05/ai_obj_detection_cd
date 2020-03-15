1. Download your own image datasets

Is is possible to use [image_downloader](https://github.com/Swaini/object_detection_retraining/blob/master/image_downloader.py) to find all related images from Google. 

2. Label images 

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

3. Label Map

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

4. Convert into TFRecords

Use conversion script to define label maps and create the `tf.Example` protocol buffer message (see example [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)) for each image/annotation/label map in the dataset.

To both create label map and TFRecords from labeled images (with train and test dataset):

```console
# Convert train folder annotation xml files to a single csv file,
# generate the `label_map.pbtxt` file to `data/annotations` directory as well.
python3 xml_to_csv.py -i label/train -o annotations/train_labels.csv -l annotations

# Convert test folder annotation xml files to a single csv.
python3 xml_to_csv.py -i test/test -o annotations/test_labels.csv

# Generate `train.record`
python3 generate_tfrecord.py --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --img_path=resized_images/train --label_map annotations/label_map.pbtxt

# Generate `test.record`
python3 generate_tfrecord.py --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --img_path=resized_images/test --label_map annotations/label_map.pbtxt
```

5. Choose the model to train and configure the training pipeline.

Use the `download_checkpoint.sh` script.

For a list of object detection model that's compatible with the Edge TPU see [here](https://coral.ai/docs/edgetpu/models-intro/).

The script also configures the `pipeline.condifg` scripts which configures the pipeline variuables (See [here](https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-training-pipeline)
):

```console
download_checkpoint.sh --network_type mobilenet_v1_ssd --train_whole_model false
```

The network_type can be either `mobilenet_v1_ssd`, or `mobilenet_v2_ssd`.l

6. Update the pipeline.config file

See: [here](https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-training-pipeline)

7. Start re-training
TODO

### TODO
* Define automatic retrieval of boxes data from xml or csv
* Add CI/CD pipeline
* Test on GoCD pipeline

### References
* [Coral Edge TPU docs](https://coral.ai/docs/edgetpu/retrain-detection/#start-training)
* [Tensorflow: using your own dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
* [Using custom dataset with Tensorflow](https://aaronjencks.blogspot.com/2019/03/using-custom-datasets-with-tensorflows.html#TFR)
* [Tensorflow: TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord)
* [Working with TFRecords](https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d)
