FROM apache/airflow:3.1.3

USER root

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean

USER airflow

RUN pip install lightgbm pyod scikit-learn==1.6.1