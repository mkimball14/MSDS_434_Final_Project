FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required system dependencies for Prophet
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose the port
EXPOSE 8080

# Run the application using entrypoint script
CMD ["./entrypoint.sh"]
