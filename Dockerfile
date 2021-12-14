FROM python:3.9.7-alpine3.14

EXPOSE 8000

WORKDIR /app

RUN apk update &&\
    apk add busybox-extras && \
    apk add bash openssh &&\
    apk add postgresql-dev gcc python3-dev #musl-dev

RUN pip install -U pip &&\
    pip install pipenv

COPY . .

RUN pipenv install --system --deploy