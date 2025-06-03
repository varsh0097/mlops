FROM tensorflow/tensorflow:2.11.0

WORKDIR /app

COPY . /app
COPY artifacts/model.lungcancercode.h5 /app/artifacts/

RUN pip install --no-cache-dir -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
