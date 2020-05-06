VERSION=$1
docker build -t rio05docker/activity_classification:convert$VERSION .
docker push rio05docker/activity_classification:convert$VERSION