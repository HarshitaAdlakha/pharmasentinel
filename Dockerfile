# PharmaSentinel Docker image
# Builds a slim container for running the Streamlit demo.
#
# Build:   docker build -t pharmasentinel .
# Run:     docker run -p 8501:8501 pharmasentinel

FROM python:3.11-slim

WORKDIR /app

# Install system deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/

# Install package
COPY setup.py .
COPY README.md .
RUN pip install --no-cache-dir -e .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
