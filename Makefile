USER   := username
REPO   := $$(basename -s .git `git remote get-url origin`)
TAG    := $$(git rev-parse --short HEAD)
IMG    := ${USER}/${REPO}:${TAG}
LATEST := ${USER}/${REPO}:latest

start:
	@docker run --name ${REPO} --mount type=bind,source="$$(pwd)",target=/usr/src/rexup -w /usr/src/rexup --gpus all -it ${LATEST} bash
	@docker container rm ${REPO}

build:
	@docker build -t ${IMG} .
	@docker tag ${IMG} ${LATEST}
