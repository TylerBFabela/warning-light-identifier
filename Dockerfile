# Small CPU-only image for the web app. Build:  docker build -t warning-light .   Run:  docker run -p 7860:7860 warning-light
FROM python:3.12-slim
WORKDIR /app

# CPU-only PyTorch (the default wheel bundles GPU libraries and is several times bigger)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY static ./static
COPY app.py meanings.json ./
COPY deploy_model ./deploy_model

# run as a normal user, not root (Hugging Face Spaces requires this too)
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]