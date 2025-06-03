FROM python:3.9

WORKDIR /app

# Copy all project files
COPY . /app

# Copy the model file explicitly
COPY artifacts/model.lungcancercode.h5 /app/artifacts/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
