FROM python:3.10

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY pyproject.toml poetry.lock* /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

COPY app/ /app/app
COPY test/ /app/test
COPY .env .env

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8080"]
