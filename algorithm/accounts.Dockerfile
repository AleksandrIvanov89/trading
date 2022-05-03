# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD src/accounts.py main.py

COPY .env .env
COPY requirements/accounts.txt requirements.txt
COPY src/libs/* libs/

RUN pip install -r requirements.txt

EXPOSE 5001

CMD [ "python", "./main.py" ]