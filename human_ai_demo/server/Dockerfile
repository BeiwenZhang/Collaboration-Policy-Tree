# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    HOST=0.0.0.0 \
    PORT=5000 \
    CONF_PATH=/app/config.json \
    PYTHONPATH=/workspace/src1:/workspace/lib/overcooked_ai

COPY human_ai_demo/server/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Use this repository's modified Overcooked runtime and Co-π-tree agent.
COPY lib/overcooked_ai /workspace/lib/overcooked_ai
RUN pip install --no-cache-dir --no-deps -e /workspace/lib/overcooked_ai
COPY src1/project_paths.py /workspace/src1/project_paths.py
COPY src1/test_Overcooked/llmtree /workspace/src1/test_Overcooked/llmtree
COPY src1/policy /workspace/src1/policy

COPY human_ai_demo/server /app
RUN cp /app/graphics/overcooked_graphics_v2.2.js /app/static/js/graphics.js \
    && mkdir -p /app/data \
    && python /app/precompute_planners.py

EXPOSE 5000
CMD ["python", "-u", "app.py"]
