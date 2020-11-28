FROM jupyter/base-notebook:python-3.8.6

WORKDIR /home/GAN_cap1

USER root

COPY ./requirements.txt .

RUN ["/bin/bash", "-c", "apt update && \
    apt install -y build-essential git && \
    pip3 install -r requirements.txt &&\
    apt remove -y git"]
