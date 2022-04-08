# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD main.py .

COPY .env .env
COPY config.json config.json
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8050

CMD [ "python", "./main.py" ]