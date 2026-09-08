FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cs_order_agent ./cs_order_agent

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

ENV PORT=8080

CMD ["sh", "-c", "uvicorn cs_order_agent.server:app --host 0.0.0.0 --port ${PORT}"]
