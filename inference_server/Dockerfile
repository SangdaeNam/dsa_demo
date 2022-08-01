ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

ENV LANG C.UTF-8
ARG PYTHON=python3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Seoul"
ENV http_proxy="http://proxy-dmz.intel.com:912"
ENV https_proxy="http://proxy-dmz.intel.com:912"
ENV no_proxy="127.0.0.1,localhost,intel.com"

WORKDIR /app
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y ninja-build && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get install -y g++ && \
    apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} \
    ${PYTHON}-pip \
    ${PYTHON}-dev 

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools \
    psutil 

EXPOSE 8501
COPY requirements.txt ./requirements.txt
COPY app.py ./app.py

RUN ${PYTHON} -m pip --no-cache-dir install -r requirements.txt
COPY . /app

WORKDIR /app
CMD ["streamlit", "run", "app.py"]