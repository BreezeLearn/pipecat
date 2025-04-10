# Base image with Python 3.11.5
FROM python:3.11.5

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    libasound2 \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code first
COPY . /app/

# Now install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    pip install "pipecat-ai[daily,openai,azure,silero,anthropic,google]"

# OpenSSL Installation (for Azure TTS or any SSL-related requirements)
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1w.tar.gz | tar zxf - && \
    cd openssl-1.1.1w && \
    ./config --prefix=/usr/local && \
    make -j $(nproc) && \
    make install_sw install_ssldirs && \
    ldconfig -v

# Environment variables
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV PYTHONUNBUFFERED=1

# Expose the FastAPI app port
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]