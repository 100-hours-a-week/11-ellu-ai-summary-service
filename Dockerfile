FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
RUN opentelemetry-bootstrap --action=install
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ./ /app/
EXPOSE 8080
CMD ["opentelemetry-instrument", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
