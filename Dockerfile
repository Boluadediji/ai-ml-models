# Use Python 3.11
FROM python:3.11-slim

WORKDIR /app

# 1. Install System Dependencies (Required for Audio/Voice)
# ffmpeg is required for m4a/wav processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements
COPY requirements.txt .

# 3. Install Python Dependencies
# (Added fastapi and uvicorn because we added main.py)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install fastapi uvicorn python-multipart

# 4. Copy Code
COPY . .

# 5. Expose Port
EXPOSE 8000

# 6. Run the App
CMD ["python", "main.py"]