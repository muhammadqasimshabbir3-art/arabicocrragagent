#!/bin/bash
# Arabic Document Intelligence Agent — Quick Start
# Assumes setup is already done (./setup.sh). Loads .env and starts services.

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/services.sh
source "${SCRIPT_DIR}/scripts/services.sh"

LANGGRAPH_PID=""
FRONTEND_PID=""

cleanup() {
    if [ -n "$LANGGRAPH_PID" ]; then
        kill "$LANGGRAPH_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo -e "${BLUE}"
echo "=================================================================="
echo "    ARABIC DOCUMENT INTELLIGENCE - QUICK START"
echo "=================================================================="
echo -e "${NC}"

# Load environment
if [ ! -f ".env" ]; then
    echo -e "${RED}.env not found. Run setup first:${NC}"
    echo "  cp .env.example .env"
    echo "  ./setup.sh"
    exit 1
fi

set -a
load_dotenv ".env"
set +a

FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}GROQ_API_KEY is not set in .env${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup...${NC}"
    ./setup.sh
    exit $?
fi

if ! uv run python -c "from agent import graph" 2>/dev/null; then
    echo -e "${YELLOW}Syncing dependencies...${NC}"
    uv sync --quiet
fi

ensure_frontend_deps() {
    if [ ! -d "frontend/node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        (cd frontend && npm install)
    fi
}

run_ui() {
    stop_frontend
    ensure_frontend_deps
    echo -e "${BLUE}Starting Document Intelligence Web UI (Vite)...${NC}"
    echo -e "${YELLOW}Open: http://localhost:${FRONTEND_PORT}${NC}"
    echo -e "${YELLOW}Tip: run ./start.sh both to start LangGraph + UI together${NC}"
    echo -e "${YELLOW}Legacy Streamlit: ./start.sh streamlit → http://localhost:${STREAMLIT_PORT}${NC}"
    echo ""
    trap - EXIT INT TERM
    (cd frontend && npm run dev -- --port "${FRONTEND_PORT}" --host 127.0.0.1)
}

run_server() {
    stop_langgraph
    echo -e "${BLUE}Starting LangGraph Server with Cloudflare Tunnel...${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Using tunnel to expose local server to smith.langchain.com${NC}"
    echo -e "${YELLOW}    (Browser security prevents direct localhost access)${NC}"
    echo ""
    echo -e "${YELLOW}After tunnel is ready, Studio will open automatically.${NC}"
    echo -e "${YELLOW}If not, visit the URL printed in the output below.${NC}"
    echo ""
    trap - EXIT INT TERM
    uv run langgraph dev --port "${LANGGRAPH_PORT}" --tunnel
}

run_both() {
    stop_all_services

    echo -e "${BLUE}Starting LangGraph Server + Document Intelligence UI...${NC}"
    echo ""
    echo -e "${YELLOW}API:    http://127.0.0.1:${LANGGRAPH_PORT}${NC}"
    echo -e "${YELLOW}UI:     http://localhost:${FRONTEND_PORT}${NC}"
    echo -e "${YELLOW}Tip: use ./start.sh server for Studio tunnel mode${NC}"
    echo ""

    # Local UI pairing: no Cloudflare tunnel (faster, avoids download hangs).
    uv run langgraph dev --port "${LANGGRAPH_PORT}" &
    LANGGRAPH_PID=$!

    echo "Waiting for LangGraph server..."
    wait_for_port "${LANGGRAPH_PORT}" 45

    ensure_frontend_deps
    trap cleanup EXIT INT TERM

    echo "Starting Document Intelligence Web UI..."
    (cd frontend && npm run dev -- --port "${FRONTEND_PORT}" --host 127.0.0.1)
}

run_streamlit() {
    stop_streamlit
    echo -e "${BLUE}Starting legacy Streamlit UI...${NC}"
    echo -e "${YELLOW}Open: http://localhost:${STREAMLIT_PORT}${NC}"
    echo -e "${YELLOW}For the modern UI use: ./start.sh ui → http://localhost:${FRONTEND_PORT}${NC}"
    echo ""
    trap - EXIT INT TERM
    uv run streamlit run streamlit_ui.py --server.port "${STREAMLIT_PORT}"
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
        ui) run_ui ;;
        server) run_server ;;
        both) run_both ;;
        streamlit) run_streamlit ;;
        *)
            echo -e "${RED}Unknown restart target: $target${NC}"
            echo "Use: ./start.sh restart [ui|server|both|streamlit]"
            exit 1
            ;;
    esac
}

show_help() {
    echo "Usage: ./start.sh [command]"
    echo ""
    echo "Commands:"
    echo "  ui              Start modern Vite/React UI (default) → :${FRONTEND_PORT}"
    echo "  server          Start LangGraph Server + Studio tunnel"
    echo "  both            Start LangGraph + modern UI"
    echo "  streamlit       Start legacy Streamlit UI → :${STREAMLIT_PORT}"
    echo "  stop            Stop LangGraph / Streamlit / Vite"
    echo "  restart [target] Restart ui, server, both, or streamlit"
    echo ""
    echo "Examples:"
    echo "  ./start.sh              # modern UI at http://localhost:${FRONTEND_PORT}"
    echo "  ./start.sh both         # API + modern UI"
    echo "  ./start.sh streamlit    # legacy UI at http://localhost:${STREAMLIT_PORT}"
    echo "  ./start.sh stop"
    echo ""
    echo "First time? Run: ./setup.sh"
    echo "Workflow docs:  AgentWorkflow.md"
}

MODE="${1:-ui}"
ARG2="${2:-}"

case "$MODE" in
    ui|frontend)
        run_ui
        ;;
    server)
        run_server
        ;;
    both)
        run_both
        ;;
    streamlit)
        run_streamlit
        ;;
    stop)
        run_stop
        ;;
    restart)
        run_restart "${ARG2:-both}"
        ;;
    -h|--help|help)
        show_help
        ;;
    "")
        run_ui
        ;;
    *)
        echo -e "${RED}Unknown option: $MODE${NC}"
        show_help
        exit 1
        ;;
esac
