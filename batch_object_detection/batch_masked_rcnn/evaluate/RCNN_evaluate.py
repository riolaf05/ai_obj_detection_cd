#Thanks to: https://databricks.com/blog/2018/09/21/how-to-use-mlflow-to-reproduce-results-and-retrain-saved-keras-ml-models.html
import keras
import mlflow
from keras.models import load_mode
import mlfow.keras

#your Keras built, trained, and tested model
model = ...
#local or remote S3 or Azure Blob path
model_dir_path=...
# save the mode to local or remote accessible path on the S3 or Azure Blob
mlflow.keras.save_model(model, model_dir_path)

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