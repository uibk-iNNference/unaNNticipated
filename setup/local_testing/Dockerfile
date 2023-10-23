FROM ubuntu:20.04


RUN apt-get update -y && \
    apt-get install -y wget sudo gnupg software-properties-common
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-dev cmake git clang

RUN adduser --gecos "" --disabled-password dev
RUN usermod -a -G sudo dev

USER dev
WORKDIR /home/dev

CMD "bash"
# CMD ["bash", "/setup/setup.sh"]