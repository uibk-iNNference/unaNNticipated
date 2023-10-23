#!/bin/bash

# create volume
docker volume create repo

# drop the repo (will be source for cloning later)
docker run --rm -i -v repo:/repo -v $HOME/Projects/forennsic:/mnt ubuntu:20.04 <<BASH

# TODO: manually copy CUDNN to /repo

cp -r /mnt /repo/forennsic
rm -rf /repo/forennsic/venv
chown -R 1000:1000 /repo

BASH
