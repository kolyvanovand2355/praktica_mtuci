# Базовый образ Python (легковесный)
FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Системные зависимости для opencv и pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements - это ускорит пересборку
COPY requirements.txt .

# Устанавливаем PyTorch CPU-версию (легче и меньше)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit

# Копируем весь проект
COPY . .

# Открываем порт Streamlit
EXPOSE 8501

# Настройки Streamlit, чтобы работал в Docker
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Запуск Streamlit-приложения
CMD ["streamlit", "run", "demo_app.py", "--server.address=0.0.0.0", "--server.port=8501"]