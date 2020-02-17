#See https://stackoverflow.com/questions/50665144/create-tfrecord-for-object-detection-task
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import csv
import json

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

LABEL_DICT = {}
counter = 0

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = 404 # Image height
  width = 720 # Image width
  filename = example['path'].encode('utf-8').strip() # Filename of the image. Empty if image is not from file

  with tf.gfile.GFile(example['path'], 'rb') as fid:
    encoded_image_data = fid.read()

  image_format = 'jpeg'.encode('utf-8').strip() # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
              # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
              # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for box in example['boxes']:
    #if box['occluded'] is False:
    #print("adding box")
    xmins.append(float(int(box['x']) / width))
    xmaxs.append(float(int(box['w']) + int(box['x']) / width))
    ymins.append(float(int(box['y']) / height))
    ymaxs.append(float(int(box['h']) + int(box['y']) / height))
    classes_text.append(box['label'].encode('utf-8'))
    classes.append(int(LABEL_DICT[box['label']]))

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example

def ex_info(img_path, ann_path):
  #TODO: code to retrieve box coordinates info
  '''
  boxes = []
  head = ['label','x','y','w','h']
  with open(ann_path, 'r') as csvfile:
    annreader = csv.DictReader(csvfile, fieldnames=head)
    for box in annreader:
      boxes.append(box)
      LABEL_DICT[box['label']] = LABEL_DICT.get(box['label'], len(LABEL_DICT) + 1)

  ex = {
    "path" : img_path,
    "boxes" : boxes
  }
  
  return ex
  '''

def main(_):
  
  # TODO(user): Write code to read in your dataset to examples variable
  dataset_dir = "dataset"
  ann_dir = join(dataset_dir, "ann")
  imgs_dir = join(dataset_dir, "imgs")
  labelDest = "tfTrain/data/labels_map.pbtxt"

  imgs = [join(imgs_dir, f) for f in listdir(imgs_dir) if isfile(join(imgs_dir, f))]
  anns = [join(ann_dir, os.path.basename(im).replace("jpg","csv")) for im in imgs]

  for img,ann in zip(imgs,anns):
    example = ex_info(img,ann)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path+example['path']+'.tfrecords')
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  '''
  with open(labelDest, 'w', encoding='utf-8') as outL:
    for name,key in LABEL_DICT.items():
      outL.write("item { \n  id: " + str(key) + "\n  name: '" + name + "'\n}\n")
  '''

  writer.close()


if __name__ == '__main__':
  tf.app.run()