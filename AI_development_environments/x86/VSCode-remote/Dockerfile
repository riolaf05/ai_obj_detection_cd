FROM ubuntu:18.04

RUN apt update && apt install  openssh-server sudo -y

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test 

RUN  echo 'test:test' | chpasswd

RUN service ssh start

EXPOSE 22

RUN apt-get update && apt-get -y update

RUN apt-get install -y build-essential python3.6 python3-pip python3-dev

RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/

#COPY requirements.txt /tmp/requirements.txt
#RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

CMD ["/usr/sbin/sshd", "&&", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]




