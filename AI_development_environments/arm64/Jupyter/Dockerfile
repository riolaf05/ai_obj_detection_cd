FROM nvcr.io/nvidia/l4t-base:r32.2.1

WORKDIR /notebooks
RUN pip install jupyter
EXPOSE 8888
RUN pip install keras

CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--notebook-dir='/notebooks'"]