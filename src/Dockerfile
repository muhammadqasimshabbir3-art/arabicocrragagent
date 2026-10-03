FROM langchain/langgraph-api:3.12

# Railway Root Directory = `src/` (build context is this folder).
# Local: docker compose build context ./src

# -- Adding local package . --
ADD . /deps/ArabicOCRRAGAgent
# -- End of local package . --

# -- Installing all local dependencies --
RUN for dep in /deps/*; do \
      echo "Installing $dep"; \
      if [ -d "$dep" ]; then \
        (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .); \
      fi; \
    done
# -- End of local dependencies install --

ENV LANGSERVE_GRAPHS='{"agent": "/deps/ArabicOCRRAGAgent/agent/graph.py:graph"}'
ENV MODELS_DIR=/deps/ArabicOCRRAGAgent/models
ENV VECTORSTORE_PERSIST_DIR=/deps/ArabicOCRRAGAgent/data/chroma
ENV OCR_ENGINE=digital

# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license \
    && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --

# -- Removing build deps from the final image --
RUN pip uninstall -y pip setuptools wheel || true
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* \
    && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* \
    && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm -f /usr/bin/uv /usr/bin/uvx || true

WORKDIR /deps/ArabicOCRRAGAgent

# Alias Railway DATABASE_URL/REDIS_URL → LangGraph DATABASE_URI/REDIS_URI
COPY deploy/railway-entrypoint.sh /deploy/railway-entrypoint.sh
RUN chmod +x /deploy/railway-entrypoint.sh

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /deps/ArabicOCRRAGAgent/models /deps/ArabicOCRRAGAgent/data/chroma \
    && chown -R appuser:appuser /deps/ArabicOCRRAGAgent
USER appuser

# Base image ENTRYPOINT is /storage/entrypoint.sh — wrap it for Railway env aliases.
ENTRYPOINT ["/deploy/railway-entrypoint.sh"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ok', timeout=3)" || exit 1
