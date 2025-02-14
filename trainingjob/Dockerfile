# Use the official Python base image with the required version
FROM python:3.12-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for scikit-survival and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY train.py .

# Set the default command to run the training script
ENTRYPOINT ["python", "train.py"]
