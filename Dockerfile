FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cs_order_agent ./cs_order_agent
COPY main.py .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

ENV PORT=8000

CMD ["python", "main.py"]
