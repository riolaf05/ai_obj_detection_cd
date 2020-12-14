FROM  arm32v7/ubuntu:16.04
COPY ./qemu-arm-static /usr/bin/qemu-arm-static

#Install Jenkins
RUN apt-get update
RUN apt-get -y --no-install-recommends install apt-transport-https ca-certificates gnupg2 software-properties-common
RUN apt-get install -y curl wget oracle-java8-jdk
RUN wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
RUN sh -c 'echo deb https://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
RUN apt-get update
RUN apt-get install -y jenkins

#Install Docker
RUN curl -sSL https://get.docker.com | bash

