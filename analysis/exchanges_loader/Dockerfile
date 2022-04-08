# Dockerfile, Image, Container
FROM python:3.8
LABEL maintainer "Aleksandr Ivanov <axeliandr@protonmail.com>"

# ENV exchange=bitfinex

ADD main.py .

COPY .env .env

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8050

CMD [ "python", "./main.py" ]