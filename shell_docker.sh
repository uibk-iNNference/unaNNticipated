#!/bin/bash

docker run --rm -it -v $HOME/Projects/forennsic:/forennsic -v $HOME/.config/gcloud:$HOME/.config/gcloud -v $HOME/.ssh:$HOME/.ssh -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK -u 1001:1001 forennsic bash
