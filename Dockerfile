FROM python:3.11-slim

WORKDIR /app
COPY requirements_v43.txt .
RUN pip install --no-cache-dir -r requirements_v43.txt

COPY . .
RUN mkdir -p /app/data

EXPOSE 8501 5000

CMD ["streamlit", "run", "app_v44.py", "--server.port=8501", "--server.address=0.0.0.0"]
