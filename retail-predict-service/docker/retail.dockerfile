FROM  python:3.10.17-slim-bullseye

# neet wget for wait-for-it bash script
RUN apt-get update && apt-get install -y wget netcat && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY config/ ./config/
COPY tasks/ ./tasks/
COPY utils/ ./utils/

COPY [ "main.py", "pyproject.toml", "./" ]