#!/bin/sh
# Map Railway's default DB/Redis names to what langchain/langgraph-api expects.
# Railway Postgres → DATABASE_URL ; LangGraph → DATABASE_URI
# Railway Redis    → REDIS_URL    ; LangGraph → REDIS_URI
set -eu

if [ -z "${DATABASE_URI:-}" ]; then
  if [ -n "${DATABASE_URL:-}" ]; then
    export DATABASE_URI="$DATABASE_URL"
  elif [ -n "${POSTGRES_URI:-}" ]; then
    export DATABASE_URI="$POSTGRES_URI"
  elif [ -n "${POSTGRES_URL:-}" ]; then
    export DATABASE_URI="$POSTGRES_URL"
  fi
fi

if [ -z "${REDIS_URI:-}" ]; then
  if [ -n "${REDIS_URL:-}" ]; then
    export REDIS_URI="$REDIS_URL"
  fi
fi

if [ -z "${DATABASE_URI:-}" ]; then
  echo "FATAL: DATABASE_URI is empty." >&2
  echo "On Railway agent service Variables, add Postgres and set ONE of:" >&2
  echo "  DATABASE_URI=\${{Postgres.DATABASE_URL}}" >&2
  echo "  DATABASE_URL=\${{Postgres.DATABASE_URL}}   (we alias this to DATABASE_URI)" >&2
  echo "Use your real Postgres *service name* inside \${{...}} (Variables → Reference)." >&2
  exit 1
fi

if [ -z "${REDIS_URI:-}" ]; then
  echo "FATAL: REDIS_URI is empty." >&2
  echo "On Railway agent service Variables, add Redis and set ONE of:" >&2
  echo "  REDIS_URI=\${{Redis.REDIS_URL}}" >&2
  echo "  REDIS_URL=\${{Redis.REDIS_URL}}   (we alias this to REDIS_URI)" >&2
  exit 1
fi

# Normalize postgres:// → postgresql:// (psycopg prefers postgresql)
case "$DATABASE_URI" in
  postgres://*)
    export DATABASE_URI="postgresql://${DATABASE_URI#postgres://}"
    ;;
esac

echo "LangGraph env OK: DATABASE_URI set, REDIS_URI set"
exec /storage/entrypoint.sh "$@"
