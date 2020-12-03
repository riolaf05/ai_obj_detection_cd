import os 
import tensorflow as tf
import tensorflow_hub as hub
import mlflow 
import argparse
import time

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def image_load(image_path):
    loaded_image = image.load_img(image_path)
    image_rel = pathlib.Path(image_path).relative_to(train_root)
    print(image_rel)
    return loaded_image

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model with unseen and untrained data
    :param model:
    :return: results of probability
    """

    return model.evaluate(x_test, y_test)

def get_binary_loss(hist):
    loss = hist.history['loss']
    loss_val = loss[len(loss) - 1]
    return loss_val

def get_binary_acc(hist):
    acc = hist.history['accuracy']
    acc_value = acc[len(acc) - 1]

    return acc_value

def get_validation_loss(hist):
    val_loss = hist.history['val_loss']
    val_loss_value = val_loss[len(val_loss) - 1]

    return val_loss_value

def get_validation_acc(hist):
    val_acc = hist.history['val_accuracy']
    val_acc_value = val_acc[len(val_acc) - 1]

    return val_acc_value


def print_metrics(hist):

    acc_value = get_binary_acc(hist)
    loss_value = get_binary_loss(hist)

    val_acc_value = get_validation_acc(hist)

    val_loss_value = get_validation_loss(hist)

    print("Final metrics: binary_loss:%6.4f" % loss_value)
    print("Final metrics: binary_accuracy=%6.4f" % acc_value)
    print("Final metrics: validation_binary_loss:%6.4f" % val_loss_value)
    print("Final metrics: validation_binary_accuracy:%6.4f" % val_acc_value)


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
    parser.add_argument('--img-size', type=int, help='Image size', default=200)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--tracking-url', type=str, help='MLFlow server')
    parser.add_argument('--epochs', type=int, help='Epochs', default=3)
    parser.add_argument('--steps-per-epoch', type=int, help='Steps per Epochs', default=21)
    parser.add_argument('--experiment', type=str, help='Experiment name', default='default')
    parser.add_argument('--version', type=str, help='Experiment version', default='latest')
    parser.add_argument('--loss', type=str, help='Loss function', default='categorical_crossentropy')
    args = parser.parse_args() 

    ### Mlflow settings
    #set MLflow server 
    mlflow.set_tracking_uri(args.tracking_url)
    #Set experiment
    if mlflow.get_experiment_by_name(args.experiment) != None:
        exp_id = mlflow.set_experiment(args.experiment)
    else: 
        exp_id = mlflow.create_experiment(args.experiment)
    
    #Close active runs
    if mlflow.active_run():
        mlflow.end_run()

    ### Train code
    
    # inser preprocess steps and model creation here

    with mlflow.start_run(run_id=None, experiment_id=exp_id, run_name=None, nested=False): 

      ### fitting the model
      
      # insert model compile here.. 
      # i.e. 
      # history = model.fit(...)

      ### mlflow autolog
      mlflow.tensorflow.autolog()

      ### Set tags
      tags={}
      tags['name']=args.experiment
      tags['version']=args.version
      mlflow.set_tags(tags)

      ### mlflow logging
      # log parameters
      mlflow.log_param("hidden_layers", args.hidden_layers)
      mlflow.log_param("output", args.output)
      mlflow.log_param("epochs", args.epochs)
      mlflow.log_param("loss_function", args.loss)
      # log metrics
      mlflow.log_metric("binary_loss", get_binary_loss(history))
      mlflow.log_metric("binary_acc",  get_binary_acc(history))
      mlflow.log_metric("validation_loss", get_binary_loss(history))
      mlflow.log_metric("validation_acc", get_validation_acc(history))
      
      ### log artifacts (matplotlib images for loss/accuracy)
      #mlflow.log_artifacts(image_dir)
      
      ### log model
      t = time.time()
      #model.save(os.path.join(BASE_DIR, "models", "{}.h5".format(int(t)))) #HDF5 format
      tf.saved_model.save(model, os.path.join(BASE_DIR, "saved_model", "{}".format(int(t)))) #SavedModel format
      mlflow.tensorflow.log_model(model, 'model')
      
      mlflow.end_run()

if __name__ == "__main__":
    main()