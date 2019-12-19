import keras
import mlflow
from keras.models import load_mode
import mlfow.keras
#your Keras built, trained, and tested model
model = load_model(model_dir_path)
with mlflow.start_run():
   # log metrics
   mlflow.log_metric("binary_loss", binary_loss)
   mlflow.log_metric("binary_acc", binary_acc)
   mlflow.log_metric("validation_loss", validation_loss)
   mlflow.log_metric("validation_acc", validation_acc)
   mlflow.log_metric("average_loss", average_loss)
   mlflow.log_metric("average_acc", average_acc)
   # log artifacts
   mlflow.log_artifacts(image_dir, "images")
   # log model
   mlflow.keras.log_model(model, "models")