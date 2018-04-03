# Development

## Build docker
```
docker build . -t aihack
```

## Run docker
```
docker run -it --rm -v $(pwd):/app -w /app aihack /bin/bash
```
