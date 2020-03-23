## Custom ML model train and deploy with MLFlow

### Installation

1. Put your train code on `train.py` script
2. Put your train and test data into `<data_folder>`
3. Run the MLFlow package to start training your model:

```console
 mlflow run . -b local --no-conda -e train
```

The Docker also contains MLFlow tracking APIs to log training metrics. See [MLFlow docs](https://www.mlflow.org/docs/latest/tracking.html).

TODO: ~~add [MLFlow packaging](https://www.mlflow.org/docs/latest/quickstart.html#running-mlflow-projects)~~ and [serving](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models). 

Otherwise, call Docker directly:

```console
docker run -it -v <data_folder>:/opt/ml/input -v <model_folder>:/opt/ml/model -p -p 8153:8153 -p 5000:5000 --rm rio05docker/obj_detection_cd:rpi3_pipeline_${GITHUB_SHA}
```

4. ~~Open GoCD UI from `localhost:8153`~~

~~Note: it is possible to get models and data from S3 or other repositories since the pipeline uses DVC~~

5. During train it is possible to monitor training through MLFlow from `localhost:5000`


### Agenda

* ~~create train Docker~~
* ~~add other MLFlow features~~
* add deploy package
* link with other ML pipeline components

### References

* [MLFlow quickstart](https://www.mlflow.org/docs/latest/quickstart.html#downloading-the-quickstart)
* [MLFlow Keras API](https://www.mlflow.org/docs/latest/python_api/mlflow.keras.html)
