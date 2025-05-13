# Use a stable base image
FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-related packages and PyTorch
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy files from the repository to the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install the latest diffusers from source (as recommended in README)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# Set Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set multiprocessing method
ENV PYTHONWARNINGS="ignore"

# Set attention provider as environment variables
ENV FINETRAINERS_ATTN_PROVIDER=native
ENV FINETRAINERS_ATTN_CHECKS=0

# Set entrypoint (use shell to allow command execution)
ENTRYPOINT ["/bin/bash"]