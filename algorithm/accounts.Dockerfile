# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

ADD src/accounts.py main.py

COPY .env .env
COPY src/accounts_requirements.txt requirements.txt
COPY src/libs/* libs/

RUN pip install -r requirements.txt

EXPOSE 5000

CMD [ "python", "./main.py" ]