FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY rlaif_lab/ ./rlaif_lab/

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["python", "-m", "rlaif_lab.server"]
