mkdir ~/Codice \
&& cd ~/Codice \
#Miniconda
&& wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh \
&& sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh \ # -> change default directory to /home/pi/miniconda3
&& echo "export PATH=/home/pi/miniconda3/bin:$PATH" >> /home/pi/.bashrc \
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
#Jupyter & Pandas
&& pip3 install jupyter \
&& pip3 install pandas \
&& sudo apt install -y python3-opencv \
#Docker
&& curl -sSL https://get.docker.com | sh \
&& sudo usermod -aG docker pi \
&& echo "export PATH=~/.local/bin:$PATH" >> /home/pi/.bashrc \
#Clean
&& sudo rm ~/Codice/Miniconda3-latest-Linux-armv7l.sh \
&& mkdir ~/Codice/notebooks \
&& sudo reboot -h now

