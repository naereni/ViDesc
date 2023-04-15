FROM ubuntu:20.04


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6


WORKDIR /videsc

COPY . /videsc/

RUN make install

CMD make run_app
