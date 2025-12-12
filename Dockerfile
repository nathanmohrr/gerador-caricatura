FROM python:3.11-slim

WORKDIR /app

# copiar apenas requirements primeiro para aproveitar cache docker
COPY requirements.txt .
# instalar dependências de sistema necessárias para OpenCV/MediaPipe
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   ffmpeg \
	   libsm6 \
	   libxext6 \
	   libgl1 \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# copiar código
COPY . .

ENV PYTHONUNBUFFERED=1

# Porta usada pelo uvicorn
EXPOSE 7860

# Permite que plataformas como Render definam a porta via variável PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
