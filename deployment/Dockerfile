FROM mambaorg/micromamba:2.10.0 AS base
WORKDIR /app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
USER root
RUN mkdir -p /data && chown "$MAMBA_USER:$MAMBA_USER" /data
USER $MAMBA_USER

FROM base AS studio
RUN micromamba install -y -n base -c conda-forge "python=3.11" "ffmpeg>=6,<8" && \
    micromamba clean --all --yes
COPY offline_ad_poc/__init__.py offline_ad_poc/job_store.py offline_ad_poc/tts_client.py /app/offline_ad_poc/
COPY demo_app/server.py /app/demo_app/server.py
COPY demo_app/web /app/demo_app/web
EXPOSE 7860
CMD ["python", "-m", "demo_app.server", "--split-workers", "--host", "0.0.0.0", "--allowed-host", "studio", "--data", "/data"]

FROM base AS tts
RUN micromamba install -y -n base -c conda-forge \
      "python=3.11" "pytorch[version='>=2.2',build='cpu_*']" "transformers>=4.49,<5" \
      safetensors "numpy>=1.26,<3" && \
    micromamba clean --all --yes
COPY offline_ad_poc/__init__.py offline_ad_poc/tts_client.py offline_ad_poc/tts_model.py offline_ad_poc/tts_service.py /app/offline_ad_poc/
EXPOSE 7861
CMD ["python", "-m", "offline_ad_poc.tts_service", "--host", "0.0.0.0"]

FROM base AS worker
RUN micromamba install -y -n base -c conda-forge \
      "python=3.11" "pytorch[version='>=2.2',build='cpu_*']" "transformers>=4.49,<5" \
      "accelerate>=1.2,<2" safetensors sentencepiece "pillow>=10,<13" \
      "numpy>=1.26,<3" "opencv>=4.10,<5" "ffmpeg>=6,<8" "llama.cpp=12418=cpu_*" && \
    micromamba clean --all --yes
COPY offline_ad_poc /app/offline_ad_poc

FROM base AS worker-cuda
COPY environment-gpu.yml /tmp/environment-gpu.yml
RUN sed -e '/^name:/d' -e 's/^  - pytorch>=2.2$/  - pytorch::pytorch=2.5.1=*cuda12.1*/' \
      /tmp/environment-gpu.yml > /tmp/worker-cuda.yml && \
    micromamba install -y -n base --file /tmp/worker-cuda.yml && \
    micromamba create -y -n llama-vl -c conda-forge "llama.cpp=12418=cpu_*" && \
    micromamba clean --all --yes
ENV PATH=/opt/conda/envs/llama-vl/bin:$PATH
COPY offline_ad_poc /app/offline_ad_poc
