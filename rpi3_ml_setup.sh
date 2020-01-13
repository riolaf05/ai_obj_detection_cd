mkdir ~/Codice \
&& cd ~/Codice \
#Miniconda
&& wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh \
&& sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh \ # -> change default directory to /home/pi/miniconda3
&& echo "export PATH=$HOME/miniconda3/bin:$PATH" >> $HOME/.bashrc \
#Tensorflow & Keras
&& sudo apt-get update \
&& sudo apt install -y libatlas-base-dev \
&& pip3 install tensorflow \
&& sudo apt-get install -y python3-numpy \
&& sudo apt-get install -y libblas-dev \
&& sudo apt-get install -y liblapack-dev \
&& sudo apt-get install -y python3-dev \
&& sudo apt-get install -y gfortran \
&& sudo apt-get install -y python3-setuptools \
&& sudo apt-get install -y python3-scipy \
&& sudo apt-get install -y python3-h5py \
&& pip3 install keras \
#Other dependencies
&& pip3 install jupyter \
&& pip3 install pandas \
&& sudo apt install -y python3-opencv \
#Docker
&& curl -sSL https://get.docker.com | sh \
&& sudo usermod -aG docker pi \
<<<<<<< HEAD
&& echo "export PATH=~/.local/bin:$PATH" >> /home/pi/.bashrc \
# Tensorflow Lite
&& wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl \
&& pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl \
=======
&& echo "export PATH=~/.local/bin:$PATH" >> $HOME/.bashrc \
>>>>>>> 6f743a92343caf1ba9c08a474d05d65285fd8ea7
#Clean
&& sudo rm ~/Codice/Miniconda3-latest-Linux-armv7l.sh \
&& mkdir ~/Codice/notebooks \
&& sudo reboot -h now