FROM tensorflow/tensorflow:2.5.0-gpu as experiments

RUN rm /etc/apt/sources.list.d/* && apt-get clean && apt-get update
RUN pip install --upgrade pip

RUN apt-get install -y cmake clang

ARG GCLOUD_FILENAME="google-cloud-cli-389.0.0-linux-x86_64.tar.gz"
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/$GCLOUD_FILENAME -O && \
    tar xf $GCLOUD_FILENAME &&  \
    ./google-cloud-sdk/install.sh && (echo "source /google-cloud-sdk/path.bash.inc" >> ~/.bashrc)
WORKDIR /innfrastructure
COPY innfrastructure .
RUN pip install -e .

WORKDIR /forennsic
COPY experiments/data data

FROM experiments as dev

WORKDIR /forennsic
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install -y ssh
ARG UID=1000
ARG GID=1000
ARG USER=dev
RUN groupadd -g $GID $USER && useradd -u $UID -g $GID -m $USER
RUN (echo "source /google-cloud-sdk/path.bash.inc || true" >> /home/$USER/.bashrc) && (echo "source /google-cloud-sdk/completion.bash.inc || true" >> /home/$USER/.bashrc)
