FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn mlflow scikit-learn pandas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
