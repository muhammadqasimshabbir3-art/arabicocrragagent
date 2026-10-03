# Security

## Secrets

- Never commit `.env`. Use Railway / Vercel secret stores.
- Do **not** expose API keys via `VITE_*` (they ship in the SPA bundle).
- Prefer CORS allowlists over browser-held bearer tokens.

## Input safety

- Filenames sanitized (`core.utils.sanitize.sanitize_filename`).
- Collection names validated.
- Upload size capped (`MAX_DOCUMENT_BYTES`, default 50 MB).

## Network

- Production: lock `CORS_ALLOW_ORIGINS` on Railway to your Vercel origin(s).
- Do **not** add `http.cors` in `langgraph.json` (overrides the env var).
- Prefer absolute `VITE_LANGGRAPH_API_URL` to the Railway host.

## Runtime

- Dockerfile runs as non-root `appuser`.
- Set `VALIDATE_SETTINGS_ON_STARTUP=true` in production.
- UI cancel is not authoritative for server-side spend.

## Data

- Uploaded documents may be sensitive; encrypt volumes where possible.
- Qdrant Cloud: TLS URLs + API keys only.
- Embedding weights under `src/models/` — do not commit secrets alongside them.
