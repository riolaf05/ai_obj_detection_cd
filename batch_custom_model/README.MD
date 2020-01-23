## Custom ML model monitoring and deploy with MLFlow

### Installation

1. Clone this repo n your local
2. Put your data into data/ folder and upgrade model definition in model/ folder
3. Push the code and wait for CI/CD pipeline to finish
4. Pull the docker image:

```console
docker run -it --rm rio05docker/obj_detection_cd:rpi3_custom_${GITHUB_SHA}
```

5. connect to localhost:5000 to see the ML workflow metadata

