# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD main.py .

COPY .env .env
COPY config.json config.json
COPY firebase_credentials.json.env firebase_credentials.json
COPY exchanges_db/* exchanges_db/
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8055

CMD [ "python", "./main.py" ]