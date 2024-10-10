USER_NAME := $(shell whoami)
IMAGE_NAME := smolseg
TAG_NAME := v0.0.1
GPU_ID := 0
CONTAINER_NAME := $(IMAGE_NAME)_container_gpu$(GPU_ID)

UID := $(shell id -u)
GID := $(shell id -g)

define run_docker
	@docker run -it --rm \
		--net host \
		--gpus '"device=$(GPU_ID)"' \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-u $(USER_NAME) \
		-v ./:/workspace \
		-v /media/robesafe/Data1/cityscapes/:/data/cityscapes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

build:
	docker build . -t $(IMAGE_NAME):$(TAG_NAME) --force-rm --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)

run:
	$(call run_docker, "bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

jupyter:
	$(call run_docker, "jupyter notebook")

stop:
	docker stop $(CONTAINER_NAME)
