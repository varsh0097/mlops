# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Ensure model is copied (if not already in the repo)
COPY artifacts/model.lungcancercode.h5 /app/artifacts/

# Install required system packages (optional but good for builds)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Disable GPU usage and reduce TensorFlow logs
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
