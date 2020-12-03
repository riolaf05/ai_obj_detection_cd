### Arm64 Docker with ML toolkit

To build:

1. enable experimental features on Docker

2. Build & Push

```console
docker buildx build --platform linux/arm64 --push -t rio05docker/ai-toolkit:arm_jupyter .
```

3. Run on Jetson board:

```console
docker run -it -p 8888:8888 -v ~/Codice/volume/jupyter:/notebooks --name jetson-jupyter rio05docker/ai-toolkit:arm_jupyter
```

### References

* https://medium.com/analytics-vidhya/deep-learning-on-jetson-nano-streamlining-docker-builds-699c8ef4a379