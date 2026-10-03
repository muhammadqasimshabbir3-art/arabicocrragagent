#!/bin/bash
# Arabic Document Intelligence — run locally (dev or prod-like)
#
# Backend: src/   Frontend: frontend/
# Requires: ./setup.sh once (or this script will sync deps).
#
# Local (dev):
#   ./start.sh both          # LangGraph + Vite HMR  ← recommended
#   ./start.sh ui            # Vite only (API must already run)
#   ./start.sh server-local  # LangGraph only (no tunnel)
#   ./start.sh server        # LangGraph + Cloudflare tunnel (Studio)
#
# Production-like (still on this machine):
#   ./start.sh build         # npm run build
#   ./start.sh prod          # build + API + vite preview (compiled UI)
#   ./start.sh docker        # docker compose up api
#
# Deploy (platforms):
#   Railway → Root Directory src
#   Vercel  → Root Directory frontend
#   See docs/DEPLOYMENT.md

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND_DIR="${SCRIPT_DIR}/src"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"

# shellcheck source=scripts/services.sh
source "${SCRIPT_DIR}/scripts/services.sh"

uv_backend() {
    uv --directory "${BACKEND_DIR}" "$@"
}

LANGGRAPH_PID=""

cleanup() {
    if [ -n "$LANGGRAPH_PID" ]; then
        kill "$LANGGRAPH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo -e "${BLUE}"
echo "=================================================================="
echo "    ARABIC DOCUMENT INTELLIGENCE - START"
echo "=================================================================="
echo -e "${NC}"

if [ ! -f ".env" ]; then
    echo -e "${RED}.env not found.${NC}"
    echo "  cp src/.env.example .env   # set GROQ_API_KEY"
    echo "  ./setup.sh"
    exit 1
fi

ensure_local_env_defaults ".env"
set -a
load_dotenv ".env"
set +a
link_backend_env "$SCRIPT_DIR" "$BACKEND_DIR"

LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
# Local UI must point at local API unless you intentionally use a remote URL.
LOCAL_API_URL="http://127.0.0.1:${LANGGRAPH_PORT}"
VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL:-$LOCAL_API_URL}"
VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID:-agent}"
GROQ_MODEL="${GROQ_MODEL:-openai/gpt-oss-120b}"

if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}GROQ_API_KEY is not set in .env${NC}"
    exit 1
fi

if [ ! -f "${BACKEND_DIR}/pyproject.toml" ]; then
    echo -e "${RED}Backend missing at src/. Is the repo complete?${NC}"
    exit 1
fi

if ! uv_backend run python -c "from agent import graph" 2>/dev/null; then
    echo -e "${YELLOW}Syncing backend dependencies...${NC}"
    uv_backend sync --quiet
fi

ensure_frontend_deps() {
    if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        (cd "${FRONTEND_DIR}" && npm install --no-audit --no-fund)
    fi
}

print_connection() {
    echo -e "${GREEN}Connected stack:${NC}"
    echo -e "  API:      ${YELLOW}${LOCAL_API_URL}${NC}"
    echo -e "  UI:       ${YELLOW}http://127.0.0.1:${FRONTEND_PORT}${NC}"
    echo -e "  Assistant:${YELLOW} ${VITE_LANGGRAPH_ASSISTANT_ID}${NC}"
    echo -e "  Model:    ${YELLOW}${GROQ_MODEL}${NC}"
    echo -e "  CORS:     includes local UI origins (see .env)"
}

start_langgraph_local() {
    stop_langgraph
    echo -e "${BLUE}Starting LangGraph API (src/) on :${LANGGRAPH_PORT}...${NC}"
    (
        cd "${BACKEND_DIR}"
        # --allow-blocking: embeddings/tool imports use sync I/O in local/dev
        uv run langgraph dev \
            --port "${LANGGRAPH_PORT}" \
            --allow-blocking \
            --no-browser \
            --host 127.0.0.1
    )
}

start_langgraph_bg() {
    stop_langgraph
    (
        cd "${BACKEND_DIR}"
        uv run langgraph dev \
            --port "${LANGGRAPH_PORT}" \
            --allow-blocking \
            --no-browser \
            --host 127.0.0.1
    ) &
    LANGGRAPH_PID=$!
    echo "Waiting for LangGraph /ok ..."
    wait_for_port "${LANGGRAPH_PORT}" 60
    if curl -fsS --max-time 2 "http://127.0.0.1:${LANGGRAPH_PORT}/ok" >/dev/null 2>&1; then
        echo -e "${GREEN}API ready at ${LOCAL_API_URL}${NC}"
    else
        echo -e "${YELLOW}API may still be starting — check logs if UI cannot connect${NC}"
    fi
}

run_ui_dev() {
    stop_frontend
    ensure_frontend_deps
    print_connection
    echo -e "${BLUE}Starting Vite DEV UI (hot reload)...${NC}"
    echo -e "${YELLOW}Tip: ./start.sh both starts API + UI together${NC}"
    echo ""
    trap - EXIT INT TERM
    (
        cd "${FRONTEND_DIR}"
        # Prefer explicit local API; Vite also proxies /api → LANGGRAPH_PORT
        VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL}" \
        VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID}" \
        npm run dev -- --port "${FRONTEND_PORT}" --host 127.0.0.1
    )
}

run_ui_preview() {
    stop_frontend
    ensure_frontend_deps
    echo -e "${BLUE}Building production frontend bundle...${NC}"
    (
        cd "${FRONTEND_DIR}"
        VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL}" \
        VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID}" \
        npm run build
    )
    print_connection
    echo -e "${BLUE}Serving compiled UI (vite preview)...${NC}"
    echo ""
    trap - EXIT INT TERM
    (
        cd "${FRONTEND_DIR}"
        npm run preview -- --host 127.0.0.1 --port "${FRONTEND_PORT}"
    )
}

run_both_dev() {
    stop_all_services
    echo -e "${BLUE}LOCAL DEV — LangGraph + React (connected)${NC}"
    echo ""
    print_connection
    echo ""
    start_langgraph_bg
    ensure_frontend_deps
    trap cleanup EXIT INT TERM
    (
        cd "${FRONTEND_DIR}"
        VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL}" \
        VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID}" \
        npm run dev -- --port "${FRONTEND_PORT}" --host 127.0.0.1
    )
}

run_prod_local() {
    stop_all_services
    echo -e "${BLUE}PROD-LIKE LOCAL — compiled UI + LangGraph${NC}"
    echo -e "${YELLOW}Same wiring as deploy: UI → VITE_LANGGRAPH_API_URL → API${NC}"
    echo ""
    # For local prod-like, always bake local API unless user set a remote URL
    case "${VITE_LANGGRAPH_API_URL}" in
        http://127.0.0.1:*|http://localhost:*|"")
            VITE_LANGGRAPH_API_URL="${LOCAL_API_URL}"
            ;;
    esac
    print_connection
    echo ""
    start_langgraph_bg
    ensure_frontend_deps
    echo -e "${BLUE}Building frontend...${NC}"
    (
        cd "${FRONTEND_DIR}"
        VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL}" \
        VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID}" \
        npm run build
    )
    trap cleanup EXIT INT TERM
    echo -e "${GREEN}Open http://127.0.0.1:${FRONTEND_PORT}${NC}"
    (
        cd "${FRONTEND_DIR}"
        npm run preview -- --host 127.0.0.1 --port "${FRONTEND_PORT}"
    )
}

run_server_tunnel() {
    stop_langgraph
    echo -e "${BLUE}Starting LangGraph + Cloudflare tunnel...${NC}"
    echo -e "${RED}SECURITY: --tunnel exposes the API publicly.${NC}"
    echo -e "${YELLOW}Prefer ./start.sh both for local UI.${NC}"
    echo ""
    trap - EXIT INT TERM
    (
        cd "${BACKEND_DIR}"
        uv run langgraph dev \
            --port "${LANGGRAPH_PORT}" \
            --allow-blocking \
            --tunnel
    )
}

run_server_local() {
    trap - EXIT INT TERM
    print_connection
    start_langgraph_local
}

run_streamlit() {
    stop_streamlit
    echo -e "${BLUE}Legacy Streamlit UI → http://localhost:${STREAMLIT_PORT}${NC}"
    echo -e "${YELLOW}Modern UI: ./start.sh both${NC}"
    echo ""
    trap - EXIT INT TERM
    uv_backend run streamlit run "${SCRIPT_DIR}/streamlit_ui.py" --server.port "${STREAMLIT_PORT}"
}

run_build() {
    ensure_frontend_deps
    echo -e "${BLUE}Building frontend (VITE_LANGGRAPH_API_URL=${VITE_LANGGRAPH_API_URL})...${NC}"
    (
        cd "${FRONTEND_DIR}"
        VITE_LANGGRAPH_API_URL="${VITE_LANGGRAPH_API_URL}" \
        VITE_LANGGRAPH_ASSISTANT_ID="${VITE_LANGGRAPH_ASSISTANT_ID}" \
        npm run build
    )
    echo -e "${GREEN}Build output: frontend/dist${NC}"
    echo -e "Deploy UI on Vercel (Root Directory = frontend) or: ./start.sh prod"
}

run_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}docker not found${NC}"
        exit 1
    fi
    echo -e "${BLUE}Starting API via docker compose (context ./src)...${NC}"
    echo -e "${YELLOW}UI separately: ./start.sh ui  (set VITE_LANGGRAPH_API_URL=http://127.0.0.1:8000)${NC}"
    trap - EXIT INT TERM
    docker compose up api --build
}

run_stop() {
    trap - EXIT INT TERM
    stop_all_services
}

run_restart() {
    local target="${1:-both}"
    run_stop
    sleep 1
    case "$target" in
        ui|frontend) run_ui_dev ;;
        server) run_server_tunnel ;;
        server-local|api) run_server_local ;;
        both|local) run_both_dev ;;
        prod|preview) run_prod_local ;;
        streamlit) run_streamlit ;;
        *)
            echo -e "${RED}Unknown restart target: $target${NC}"
            show_help
            exit 1
            ;;
    esac
}

show_help() {
    cat <<EOF
Usage: ./start.sh [command]

Local development (hot reload):
  both | local      LangGraph (:${LANGGRAPH_PORT}) + Vite UI (:${FRONTEND_PORT})  [recommended]
  ui | frontend     Vite UI only
  server-local|api  LangGraph only (no tunnel)
  server            LangGraph + Cloudflare tunnel (Studio)
  streamlit         Legacy Streamlit UI (:${STREAMLIT_PORT})

Production-like on this machine:
  build             npm run build → frontend/dist
  prod | preview    Build UI + API + vite preview (compiled)
  docker            docker compose up api  (src/Dockerfile)

Ops:
  stop              Stop API / UI / Streamlit
  restart [target]  Restart a mode above
  help              This message

Deploy (cloud):
  Railway  → Root Directory = src     (see src/deploy/README.md)
  Vercel   → Root Directory = frontend (frontend/vercel.json)
  Set VITE_LANGGRAPH_API_URL to the Railway HTTPS URL on Vercel.

First time: ./setup.sh
EOF
}

MODE="${1:-both}"
ARG2="${2:-}"

case "$MODE" in
    ui|frontend) run_ui_dev ;;
    both|local) run_both_dev ;;
    prod|preview) run_prod_local ;;
    build) run_build ;;
    docker) run_docker ;;
    server) run_server_tunnel ;;
    server-local|api) run_server_local ;;
    streamlit) run_streamlit ;;
    stop) run_stop ;;
    restart) run_restart "${ARG2:-both}" ;;
    -h|--help|help) show_help ;;
    *)
        echo -e "${RED}Unknown option: $MODE${NC}"
        show_help
        exit 1
        ;;
esac
