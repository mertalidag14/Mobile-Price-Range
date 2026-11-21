FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py logolar.py gercek_telefonlar.csv ./
COPY .streamlit ./.streamlit

# Render, uygulamanın dinlemesi gereken portu PORT değişkeniyle verir
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
