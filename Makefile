all: build

build: Dockerfile .dockerignore
	docker build -t forennsic --target experiments .

publish: publish_gcp publish_aws

publish_gcp: build
	docker image tag forennsic eu.gcr.io/forennsic/experiments
	docker image push eu.gcr.io/forennsic/experiments
	
publish_aws: build
	docker image tag forennsic 457863446379.dkr.ecr.eu-central-1.amazonaws.com/forennsic:latest
	docker image push 457863446379.dkr.ecr.eu-central-1.amazonaws.com/forennsic:latest

dev: Dockerfile .dockerignore
	 docker build --target dev -t forennsic --build-arg GID=$(shell id -g) --build-arg UID=$(shell id -u) --build-arg USER=$(shell whoami) .
