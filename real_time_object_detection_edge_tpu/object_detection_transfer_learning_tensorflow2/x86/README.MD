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