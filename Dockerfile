FROM python:3.11-alpine
RUN apk update
RUN apk add py-pip
RUN apk add --no-cache python3-dev
WORKDIR /app
COPY . /app
RUN pip --no-cache-dir install -r requirements.txt
CMD ["python3", "app.py"]