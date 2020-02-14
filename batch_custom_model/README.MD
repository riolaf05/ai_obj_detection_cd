## Custom ML model monitoring and deploy with MLFlow

### Installation

1. Put your train code on `train.py` script
2. Push the code and wait for CI/CD pipeline to finish
3. Run the Docker container to start training your model:

```console
docker run -it -v <data_folder>:/opt/ml/input -v <model_folder>:/opt/ml/model -p -p 8153:8153 -p 5000:5000 --rm rio05docker/obj_detection_cd:rpi3_pipeline_${GITHUB_SHA}
```

4. Open GoCD UI from `localhost:8153`

Note: it is possible to get models and data from S3 or other repositories since the pipeline uses DVC

5. During train it is possible to monitor training through MLFlow from `localhost:5000`

### Local installation

```console
docker run -it --rm rio05docker/obj_detection_cd:rpi3_custom_${GITHUB_SHA}
```


