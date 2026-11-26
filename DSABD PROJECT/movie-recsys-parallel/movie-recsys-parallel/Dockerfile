FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ make libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN make train            # compiles the OpenMP core and trains the model into /app/artifacts
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
