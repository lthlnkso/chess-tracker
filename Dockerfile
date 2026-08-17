# Serving image for the "Who are you?" demo.
#
# CPU-only torch on purpose. The default torch wheel drags in the whole CUDA
# stack (~2.5 GB) which this never uses -- inference is a handful of forward
# passes and a 558k x 128 matmul. The cpu index brings the image down by an
# order of magnitude and makes the deploy fit a small host.
FROM python:3.12-slim

# Model weights and the gallery are NOT in git (266 MB, and .npz/.pt are
# ignored). They are fetched at BUILD time so the running container starts fast
# and a restart cannot fail on a network hiccup -- see scripts/fetch_artifacts.sh.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# torch first and from the CPU index, so the layer caches independently of the
# application code -- rebuilding for a code change should not re-download it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
RUN pip install "numpy>=1.26" "python-chess>=1.11"

COPY . /app

# ARTIFACT_BASE is where the three big files live. Defaults to the GitHub
# Release; override for S3 or any static host.
ARG ARTIFACT_BASE=""
RUN ARTIFACT_BASE="$ARTIFACT_BASE" bash scripts/fetch_artifacts.sh \
    && python -c "import os; \
        assert os.path.getsize('play/gallery_ctx10.npz') > 100_000_000, 'gallery missing'; \
        print('artifacts present')"

EXPOSE 8000

# Health check hits /healthz, which reports whether the gallery actually loaded
# rather than merely whether the socket is open.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# Flags are explicit rather than relying on argparse defaults: those defaults
# have been edited more than once, and a deploy must not silently change model.
CMD ["sh", "-c", "python play/server.py \
     --ckpt ckpt/final/ctx5_pre.pt \
     --id-ckpt ckpt/final/ctx10_ft.pt \
     --gallery play/gallery_ctx10.npz \
     --verifier '' --bayes '' \
     --host \"$HOST\" --port \"$PORT\""]
