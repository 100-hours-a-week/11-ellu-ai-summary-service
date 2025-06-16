FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
RUN opentelemetry-bootstrap --action=install
WORKDIR /app
COPY ./ /app/
EXPOSE 8080
<<<<<<< HEAD
CMD ["opentelemetry-instrument", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
=======
CMD ["opentelemetry-instrument", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
>>>>>>> c014e8772082cf46386b43e5b6582b88227e4d14
