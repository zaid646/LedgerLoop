# Minimal Ray image for the ingestion benchmark.
#
# The benchmark worker only needs the extraction/policy/embedding modules, so we
# install Ray plus pydantic and copy the code instead of syncing every service
# dependency. The repo is also bind-mounted at /workspace by docker-compose so
# edits are picked up without rebuilding.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/ai-service:/workspace/evals:/workspace/scripts

RUN pip install --no-cache-dir \
    "ray[default]==2.44.1" \
    "pydantic>=2.8" \
    "pydantic-settings>=2.4"

WORKDIR /workspace
COPY ai-service /workspace/ai-service
COPY evals /workspace/evals
COPY scripts /workspace/scripts

CMD ["bash"]
