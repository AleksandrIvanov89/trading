# Dockerfile, Image, Container
FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD main.py .
COPY requirements.txt requirements.txt
COPY config.json config.json
COPY include/ta_cuda.py include/ta_cuda.py

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt

CMD [ "python3", "./main.py" ]