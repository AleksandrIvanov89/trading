# Dockerfile, Image, Container
FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD main.py .
COPY requirements.txt requirements.txt
COPY config.json config.json
#COPY tacuda/ta_cuda.py tacuda/ta_cuda.py
#COPY tacuda/ta_functions.py tacuda/ta_functions.py
#COPY tacuda/__init__.py tacuda/__init__.py
COPY tacuda/* tacuda/

RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt

CMD [ "python3", "./main.py" ]