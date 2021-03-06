### Docker for Jetson Nano

Use [Nvidia NGC provided Docker](https://ngc.nvidia.com/catalog/containers).

For ML Container use:

```console
mkdir -p $HOME/Codice/notebooks
docker run -it --restart=unless-stopped -d -v $HOME/Codice/notebooks:/notebooks --runtime nvidia --network host nvcr.io/nvidia/l4t-ml:r32.5.0-py3
```console


