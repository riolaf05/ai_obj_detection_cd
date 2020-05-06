VERSION=$1
docker build -t rio05docker/activity_classification:preprocess$VERSION .
docker push rio05docker/activity_classification:preprocess$VERSION