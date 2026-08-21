FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn prodml.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
