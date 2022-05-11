# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD src/algo.py main.py

COPY .env .env
COPY requirements/algo.txt requirements.txt
COPY src/libs/* libs/

RUN pip install -r requirements.txt

EXPOSE 8060

CMD [ "python", "./main.py" ]