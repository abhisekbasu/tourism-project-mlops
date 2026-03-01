FROM python:3.12-slim
WORKDIR /app

# Installing git because huggingface_hub needs it to download model files
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

# Exposing port 7860 which is the default port for Hugging Face Spaces
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
