# Dockerfile for ingesting Sentinel-2 data.

FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime@sha256:d15e9803095e462e351f097fb1f5e7cdaa4f5e855d7ff6d6f36ec4c2aa2938ea

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget
RUN git clone https://github.com/allenai/rslearn.git /opt/rslearn
RUN pip install /opt/rslearn[extra]

COPY . /opt/helios/
WORKDIR /opt/helios/
