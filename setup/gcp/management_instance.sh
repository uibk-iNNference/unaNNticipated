#!/bin/bash

gcloud compute instances create setup \
    --project=forennsic --zone=europe-west1-b \
    --machine-type=n1-standard-2 \
    --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible \
    --service-account=forennsic@forennsic.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image-family=forennsic-gpu-experiment \
    --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
