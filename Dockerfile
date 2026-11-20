# Experiment environment (Python 3.11). Dependencies are listed in requirements.txt.
#
#   docker compose build
#   docker compose run --rm app python experiments/<script>.py
#   docker compose up jupyter        # http://127.0.0.1:8888 (token in the logs)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["bash"]
