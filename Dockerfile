# Image for Argo/Kubernetes steps: `sciflow run-step` against a store mounted at /store.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# Install exactly the versions in uv.lock, using the image's Python.
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --no-editable --no-cache
ENV PATH="/app/.venv/bin:$PATH" SCIFLOW_STORE=/store
ENTRYPOINT []
CMD ["sciflow", "--help"]
