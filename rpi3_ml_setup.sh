mkdir ~/Codice \
&& cd ~/Codice \
&& wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh \
&& sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh \ # -> change default directory to /home/pi/miniconda3
&& sudo echo export PATH="/home/pi/miniconda3/bin:$PATH" >> /home/pi/.bashrc \
&& sudo apt-get update \
&& sudo apt install -y libatlas-base-dev \
&& pip3 install tensorflow \
&& sudo apt-get install -y python3-numpy \
&& sudo apt-get install -y libblas-dev \
&& sudo apt-get install -y liblapack-dev \
&& sudo apt-get install -y python3-dev \
&& sudo apt-get install -y libatlas-base-dev \
&& sudo apt-get install -y gfortran \
&& sudo apt-get install -y python3-setuptools \
&& sudo apt-get install -y python3-scipy \
&& sudo apt-get install -y python3-h5py \
&& pip3 install keras \
&& pip3 install jupyter \
&& sudo apt install -y python3-opencv \
&& curl -sSL https://get.docker.com | sh \
&& sudo usermod -aG docker pi \
&& export PATH=~/.local/bin/jupyter-notebook:$PATH \
&& sudo rm ~/Codice/Miniconda3-latest-Linux-armv7l.sh \
&& mkdir ~/Codice/notebooks \
&& sudo reboot -h now

