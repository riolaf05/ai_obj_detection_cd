VERSION=$1
docker build -t rio05docker/chatbot:$VERSION .
docker push rio05docker/chatbot:$VERSION

