FROM python:3.11-slim

WORKDIR /app

# copiar apenas requirements primeiro para aproveitar cache docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copiar código
COPY . .

ENV PYTHONUNBUFFERED=1

# Porta usada pelo uvicorn
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
