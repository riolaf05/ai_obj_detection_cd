1. Download your own image datasets

Is is possible to use [image_downloader](https://github.com/Swaini/object_detection_retraining/blob/master/image_downloader.py) to find all related images from Google. 

2. Label images 

This steps produces a list of **bounding boxes** for the image. You can use [Labelimg](https://github.com/tzutalin/labelImg) for annotations. Labelimg is a great tool for drawing bounding boxes on images, the bounding boxes get automatically converted into xml documents which you can then use for tensorflow.

e.g.:

```xml
<annotation>
    <folder>OXIIIT</folder>
    <filename>Abyssinian_1.jpg</filename>
    <source>
        <database>OXFORD-IIIT Pet Dataset</database>
        <annotation>OXIIIT</annotation>
        <image>flickr</image>
    </source>
    <size>
        <width>600</width>
        <height>400</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>cat</name>
        <pose>Frontal</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <bndbox>
            <xmin>333</xmin>
            <ymin>72</ymin>
            <xmax>425</xmax>
            <ymax>158</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
</annotation>
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

call it `label_map.pbtxt`

4. Convert into TFRecords

Use conversion script to define label maps is `conversion_script.py` to create the `tf.Example` protocol buffer message (see example [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)) for each image/annotation/label map in the dataset.

```python
python3 conversion_script.py --output_path=<path> --label_map_path=<label_map>
```

5. Choose the model to train and configure the training pipeline.

Use the `download_checkpoint.sh` script.

For a list of object detection model that's compatible with the Edge TPU see [here](https://coral.ai/docs/edgetpu/models-intro/).

The script also configures the `pipeline.condifg` scripts which configures the pipeline variuables (See [here](https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-training-pipeline)
). 

6. Start re-training
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
