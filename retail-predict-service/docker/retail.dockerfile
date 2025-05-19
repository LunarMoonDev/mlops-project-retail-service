FROM  python:3.10.17-slim-bullseye

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY config/ ./config/
COPY tasks/ ./tasks/
COPY utils/ ./utils/

COPY [ "main.py", "pyproject.toml", "./" ]

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
