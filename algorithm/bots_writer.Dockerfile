# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD src/bots_writer.py main.py

COPY .env .env
# COPY firebase_credentials.json.env firebase_credentials.json
COPY requirements/writer.txt requirements.txt
COPY src/libs/* libs/

RUN pip install -r requirements.txt

EXPOSE 8055

CMD [ "python", "./main.py" ]