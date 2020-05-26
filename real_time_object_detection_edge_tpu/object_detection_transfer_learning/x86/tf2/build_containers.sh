VERSION=$1
docker build -t rio05docker/tf2_transfer_learning:$VERSION
docker push rio05docker/tf2_transfer_learning:$VERSION