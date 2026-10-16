# Hugging Face Space for the Streamlit demo (https://huggingface.co/docs/hub/spaces-sdks-docker).
# It clones the project from GitHub and downloads the trained model from the GitHub Release,
# so the Space repo only needs this Dockerfile and its README.md.
# To pick up new code or a new model, restart the Space with "Factory reboot".
FROM python:3.13-slim

ARG REPO_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: OpenMP runtime for LightGBM. git: to clone the project.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    READMISSION_MODEL_PATH=/home/user/app/models/model.joblib
WORKDIR $HOME

RUN git clone --depth 1 --branch ${REPO_REF} https://github.com/agilmouzo/readmission-risk.git app
WORKDIR $HOME/app

# Same pinned library versions the model was saved with.
RUN pip install -c constraints.txt ".[demo]" \
    && python scripts/download_model.py

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
