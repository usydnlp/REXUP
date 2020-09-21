FROM tensorflow/tensorflow:1.14.0-gpu-py3
WORKDIR /usr/src/rexup
COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]
