# ---- Base Python ----
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System deps for OCR ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir & deps ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# ---- Copy app ----
COPY . /app

# Optional env (code already reads these)
ENV POPPLER_PATH=/usr/bin \
    TESSERACT_CMD=/usr/bin/tesseract

# Azure will inject PORT; local default 8000
EXPOSE 8000
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0"]
