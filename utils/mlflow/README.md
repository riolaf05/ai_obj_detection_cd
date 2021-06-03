docker build -t rio05docker/ai-toolkit:x86_mlflow .
docker run -it --restart=unless-stopped --name mlflow-ui rio05docker/ai-toolkit:x86_mlflow 