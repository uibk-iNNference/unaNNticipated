#!/bin/bash

gcloud compute instances create setup \
    --project=forennsic --zone=europe-west4-a \
    --machine-type=n1-standard-2 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible \
    --service-account=forennsic@forennsic.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=setup,image=projects/cos-cloud/global/images/cos-85-13310-1453-16,mode=rw,size=20
