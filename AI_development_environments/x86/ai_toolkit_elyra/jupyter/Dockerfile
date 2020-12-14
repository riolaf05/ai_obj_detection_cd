FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3-pip python3-dev
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]