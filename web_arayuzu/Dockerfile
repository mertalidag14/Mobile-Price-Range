FROM python:3.13-slim

WORKDIR /app

# XGBoost'un çalışması için gereken OpenMP kütüphanesi
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py logolar.py gercek_telefonlar.csv ./
COPY .streamlit ./.streamlit
COPY fontlar ./fontlar

# Render, uygulamanın dinlemesi gereken portu PORT değişkeniyle verir
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
