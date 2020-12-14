chmod +x qemu-arm-static
docker build -t rio05docker/obj_detection_cd:jenkins_armv7 .
docker push rio05docker/obj_detection_cd:jenkins_armv7

#docker run -it -p 8080:8080 -p 50000:50000 -v /home/rosario/Codice/jenkins-arm32v7/jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock --restart unless-stopped --name jenkins_armv7 rio05docker/obj_detection_cd:jenkins_armv7