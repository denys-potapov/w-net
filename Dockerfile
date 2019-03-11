FROM tensorflow/tensorflow:1.1.0

RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1
ENV MPLBACKEND=agg

COPY . .