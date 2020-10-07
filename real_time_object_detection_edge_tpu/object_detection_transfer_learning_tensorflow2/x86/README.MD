Thank to [Transfer Learning in Deep Learning Using Tensorflow 2.0](https://www.pluralsight.com/guides/transfer-learning-in-deep-learning-using-tensorflow-2.0)

### Info

This is a transfer learning demo which uses Tensorflow 2.0 and MlFlow for  MLOps pipeline management. 

The model is a MobileNet architecture from **Tensorflow Hub** re-trained on custom images.

Example:

```console
python main.py python main.py --tracking-url <mlflow_server_url> --epochs 1 --experiment alien-vs-predator --version 10
```

### Next steps:

1. export model to SavedModel format
2. convert into Tensorflow Lite
3. Complete pipeline

### Notes:

1. Keras artifacts and log artifacts must be saved into the same forder read by the [MlFlowUI](https://github.com/riolaf05/continuous-learning-platform/tree/master/mlflowUI) (*/tmp/mlflowruns/1* by default), otherwise they wont be showed on the interface.

### References

1. [Keras model with MlFlow](https://databricks.com/blog/2018/08/23/how-to-use-mlflow-to-experiment-a-keras-network-model-binary-classification-for-movie-reviews.html)