FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy PATH=/app/.venv/bin:$PATH
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group ui --no-install-project

COPY ui ./ui
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
