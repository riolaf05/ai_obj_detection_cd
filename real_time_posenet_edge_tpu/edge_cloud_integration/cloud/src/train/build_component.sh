GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
DT=$(date '+%d-%m-%Y')
echo rio05docker/activity_classification:train-${GIT_BRANCH}-${DT}
docker build -t rio05docker/activity_classification:train_${GIT_BRANCH}_${DT} .
docker push rio05docker/activity_classification:train_${GIT_BRANCH}_${DT}
