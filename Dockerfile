FROM python:3.9-slim

# Avoid writing .pyc files and force logs to flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project into the container
COPY . /app/

EXPOSE 5000

CMD ["python", "app.py"]
