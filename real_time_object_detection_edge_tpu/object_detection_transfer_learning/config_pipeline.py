import os
import shutil
import glob
import urllib.request
import tarfile
import argparse
import re
sys.path.insert(1, '/tensorflow')
from models.research.object_detection.utils import label_map_util

BASE_DIR='/tensorflow'

# Number of training steps.
num_steps = 1000  # 200000

# Number of evaluation steps.
num_eval_steps = 50

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    }
}

def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

def main():

    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('--base-model', type=str, help='Base Model', default='ssd_mobilenet_v2')
    args = parser.parse_args()

    # Pick the model you want to use
    # Select a model in `MODELS_CONFIG`.
    selected_model = args.base_model 

    # Name of the object detection model to use.
    MODEL = MODELS_CONFIG[selected_model]['model_name']
    # Name of the pipline file in tensorflow object detection API.
    pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
    # Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
    batch_size = MODELS_CONFIG[selected_model]['batch_size']

    #Download base model file
    MODEL_FILE = MODEL + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    DEST_DIR = os.path.join(BASE_DIR, '/models/research/pretrained_model')
    if not (os.path.exists(MODEL_FILE)):
        urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()
    os.remove(MODEL_FILE)
    if (os.path.exists(DEST_DIR)):
        shutil.rmtree(DEST_DIR)
    os.rename(MODEL, DEST_DIR)

    #Get config file
    pipeline_fname = os.path.join(BASE_DIR, 'models/research/object_detection/samples/configs/', pipeline_file)
    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
    
    #Select checkpoint file
    fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")

    #Select new training data
    test_record_fname = os.path.join(BASE_DIR, 'data/annotations/test.record')
    train_record_fname = os.path.join(BASE_DIR, 'data/annotalstions/train.record')
    label_map_pbtxt_fname = os.path.join(BASE_DIR, 'data/annotations/label_map.pbtxt')

    num_classes = get_num_classes(label_map_pbtxt_fname)

    #Edit config file
    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        
        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)
        
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)
        f.write(s)

    model_dir = BASE_DIR+'training/'
    os.makedirs(model_dir, exist_ok=True)
    print(pipeline_fname)

if __name__ == "__main__":
    main()