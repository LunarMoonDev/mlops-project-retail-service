FROM python:3.10-slim

RUN pip install mlflow==2.12.1
RUN pip install boto3
RUN pip install psycopg2-binary
RUN pip install pymysql

EXPOSE 5000