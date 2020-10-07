import os 
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import mlflow 
import argparse
import time

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASE_DIR=r"C:\Users\lafacero\Documents\GitHub\ai_obj_detection_cd\real_time_object_detection_edge_tpu\object_detection_transfer_learning_tensorflow2\x86"
train_root = BASE_DIR+r"\archive\data\train"
test_root = BASE_DIR+r"\archive\data\validation"
feature_extractor_url = r"https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"


def image_load(image_path):
    loaded_image = image.load_img(image_path)
    image_rel = pathlib.Path(image_path).relative_to(train_root)
    print(image_rel)
    return loaded_image

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

#create a custom callback to visualize the training progress during every epoch.
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['accuracy'])

def main():
    parser = argparse.ArgumentParser(description='Input arguments')
    #parser.add_argument('--img-size', type=int, help='Image size', default=200)
    #parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--tracking-url', type=str, help='MLFlow server')
    parser.add_argument('--epochs', type=int, help='Epochs', default=3)
    parser.add_argument('--steps-per-epoch', type=int, help='Steps per Epochs', default=21)
    parser.add_argument('--experiment', type=str, help='Experiment name', default='default')
    parser.add_argument('--version', type=str, help='Experiment version', default='latest')
    args = parser.parse_args() 

    #Mlflow settings
    #set MLflow server 
    mlflow.set_tracking_uri(args.tracking_url)
    #Set experiment
    if mlflow.get_experiment_by_name(args.experiment) != None:
        exp_id = mlflow.set_experiment(args.experiment)
    else: 
        exp_id = mlflow.create_experiment(args.experiment)
    #Set tags
    tags={}
    tags['name']=args.experiment
    tags['version']=args.version
    mlflow.set_tags(tags)
    #Close active runs
    if mlflow.active_run():
        mlflow.end_run()

    #Train code
    train_generator = ImageDataGenerator(rescale=1/255) 
    test_generator = ImageDataGenerator(rescale=1/255) 

    train_image_data = train_generator.flow_from_directory(str(train_root),target_size=(224,224))
    test_image_data = test_generator.flow_from_directory(str(test_root), target_size=(224,224))

    IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
    print("Image size ", IMAGE_SIZE)

    for image_batch, label_batch in train_image_data:
        print("Image-batch-shape:",image_batch.shape)
        print("Label-batch-shape:",label_batch.shape)
        break

    for test_image_batch, test_label_batch in test_image_data:
        print("Image-batch-shape:",test_image_batch.shape)
        print("Label-batch-shape:",test_label_batch.shape)
        break

    feature_extractor_layer = layers.Lambda(feature_extractor,input_shape=IMAGE_SIZE+[3])
    feature_extractor_layer.trainable = False

    model = Sequential([
        feature_extractor_layer,
        layers.Dense(train_image_data.num_classes, activation = "softmax")
        ])
    model.summary()

    # initialize the TFHub module
    sess = K.get_session() 
    init = tf.global_variables_initializer()
    sess.run(init)

    model.compile(
        optimizer = tf.train.AdamOptimizer(),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
        )

    # Early stopping to stop the training if loss start to increase. It also avoids overvitting.
    es = EarlyStopping(patience=2,monitor="val_loss")

    #use CallBacks to record accuracy and loss.
    batch_stats = CollectBatchStats()

    with mlflow.start_run(run_id=None, experiment_id=exp_id, run_name=None, nested=False): 
      # fitting the model
      model.fit((item for item in train_image_data), epochs = args.epochs,
              steps_per_epoch=21,
              callbacks = [batch_stats, es],validation_data=test_image_data)

      #print labels
      label_names = sorted(train_image_data.class_indices.items(), key=lambda pair:pair[1])
      label_names = np.array([key.title() for key, value in label_names])
      print(label_names)

      #test results
      result_batch = model.predict(test_image_batch)
      labels_batch = label_names[np.argmax(result_batch, axis=-1)]
      print(labels_batch)

      mlflow.tensorflow.autolog()

      #save model 
      t = time.time()
      export_path = os.path.join(BASE_DIR, "models"+"{}".format(int(t)))
      model.save(export_path)
      mlflow.tensorflow.log_model(model, "model", registered_model_name=args.expertiment+"_"+args.version) 

      mlflow.end_run()

if __name__ == "__main__":
    main()