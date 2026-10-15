# Skill-Match Copilot API + UI image (CPU). Build:  docker build -t skill-match-copilot .
# Run API:  docker run -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models -v $(pwd)/chroma:/app/chroma skill-match-copilot
# Run UI:   docker run -p 8501:8501 --env-file .env -v ... skill-match-copilot streamlit run src/ui/app.py --server.address 0.0.0.0
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt \
 && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY configs ./configs
COPY prompts ./prompts
COPY src ./src
COPY data/demo ./data/demo
COPY data/raw/github ./data/raw/github
COPY results/p2/nil_threshold_sweep.csv.meta.json ./results/p2/
# models/ and chroma/ are mounted at run time (see docs/architecture.md)

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
