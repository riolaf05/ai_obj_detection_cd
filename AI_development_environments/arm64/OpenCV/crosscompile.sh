docker buildx build --platform linux/arm64 -t rio05docker/ai-toolkit:arm64_opencv . \
&& docker push rio05docker/ai-toolkit:arm64_opencv