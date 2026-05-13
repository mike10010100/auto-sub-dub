# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    cmake \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyphen rvc-python

# Clone and build s2.cpp with CUDA support (using parallel cores for speed)
RUN git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git \
    && cd s2.cpp \
    && cmake -B build -DS2_CUDA=ON \
    && cmake --build build --config Release --parallel $(nproc)

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models/rvc output/audio_segments output/references

# Download Fish Speech models during build
RUN python scripts/download_fish_models.py

# Expose Streamlit port
EXPOSE 8501

# Start the web interface by default
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
