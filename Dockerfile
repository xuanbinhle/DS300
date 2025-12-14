FROM apache/spark:3.5.6-scala2.12-java11-python3-ubuntu

USER root
RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        python3-pip \
        python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Set Vietnam timezone
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements and install Python packages
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt

ENV HF_HOME=/app/hf_cache
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

COPY /jars/spark-sql-kafka-0-10_2.12-3.5.6.jar /opt/spark/jars/
COPY /jars/spark-token-provider-kafka-0-10_2.12-3.5.6.jar /opt/spark/jars/
COPY /jars/commons-pool2-2.11.1.jar /opt/spark/jars/
COPY /jars/spark-excel_2.12-3.5.1_0.20.4.jar /opt/spark/jars/