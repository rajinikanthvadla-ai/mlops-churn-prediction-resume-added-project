# mlflow/Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.10.2 \
    boto3==1.34.34 \
    pymysql==1.1.0 \
    sqlalchemy==2.0.25

COPY setup_mlflow.sh /setup_mlflow.sh
RUN chmod +x /setup_mlflow.sh

EXPOSE 5000

CMD ["/setup_mlflow.sh"]
