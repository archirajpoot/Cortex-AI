# ──────────────────────────────────────────────────────────────────────────────
# CustomerSupportEnv — Hugging Face Spaces Dockerfile
# Runs the combined FastAPI backend + Gradio dashboard on port 7860 via run.py
# Build: docker build -t customer-support-env .
# Run:   docker run -p 7860:7860 customer-support-env
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Copy dependency manifests first (layer-cache friendly) ───────────────────
COPY pyproject.toml ./
# uv.lock is optional — use it if present for reproducible builds
COPY uv.lock* ./

# ── Install all project dependencies (no project install yet) ────────────────
RUN uv sync --no-install-project --no-editable 2>/dev/null || \
    uv sync --no-install-project

# ── Copy full source ─────────────────────────────────────────────────────────
COPY . .

# ── Install the project itself (editable) ───────────────────────────────────
RUN uv sync --no-editable 2>/dev/null || uv sync

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
# Tell HF Spaces which port to expose
ENV PORT=7860

# ── HF Space: run on 0.0.0.0 so the proxy can reach it ──────────────────────
# We override the host in run.py via this env var
ENV HOST=0.0.0.0

EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "run.py"]
