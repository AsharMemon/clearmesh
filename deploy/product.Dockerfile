FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-product.txt ./
RUN pip install --no-cache-dir -r requirements-product.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "clearmesh.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
