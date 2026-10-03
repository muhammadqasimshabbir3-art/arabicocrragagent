#!/bin/bash
# Arabic Document Intelligence — one-time setup (local + deploy prep)
#
# Backend: src/   Frontend: frontend/
#
# Usage:
#   ./setup.sh              # interactive menu after install
#   ./setup.sh --skip-run   # install/verify only
#   ./setup.sh both         # install then start API + React UI (local)
#   ./setup.sh prod         # install, build UI, start API + preview (prod-like)
#   ./setup.sh help

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

SKIP_RUN=0
POST_ACTION=""

case "${1:-}" in
    --skip-run|skip-run) SKIP_RUN=1 ;;
    both|local|ui|server|streamlit|prod|preview|docker) POST_ACTION="$1" ;;
    -h|--help|help)
        echo "Usage: ./setup.sh [--skip-run|both|prod|ui|server|streamlit|docker]"
        echo ""
        echo "  (no args)   Install + interactive menu"
        echo "  --skip-run  Install/verify only"
        echo "  both|local  Install then ./start.sh both"
        echo "  prod        Install, build UI, then ./start.sh prod"
        echo "  docker      Install hints + docker compose up api"
        exit 0
        ;;
    "") ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Try: ./setup.sh help"
        exit 1
        ;;
esac

echo -e "${BLUE}"
echo "=================================================================="
echo "       ARABIC DOCUMENT INTELLIGENCE - SETUP"
echo "       Backend: src/   ·   Frontend: frontend/"
echo "=================================================================="
echo -e "${NC}"

# ─── 1. Environment ──────────────────────────────────────────────────────────
echo -e "${YELLOW}Step 1: Checking environment...${NC}"

if [ ! -f ".env" ]; then
    if [ -f "${BACKEND_DIR}/.env.example" ]; then
        echo -e "${YELLOW}Creating .env from src/.env.example...${NC}"
        cp "${BACKEND_DIR}/.env.example" .env
    elif [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo -e "${RED}Error: no .env.example found${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Add GROQ_API_KEY to .env, then re-run ./setup.sh${NC}"
fi

ensure_local_env_defaults ".env"
set -a
load_dotenv ".env"
set +a
link_backend_env "$SCRIPT_DIR" "$BACKEND_DIR"

LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
GROQ_MODEL="${GROQ_MODEL:-openai/gpt-oss-120b}"

if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}Error: GROQ_API_KEY not set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}Environment check passed${NC}"
echo -e "   GROQ_API_KEY: set"
echo -e "   GROQ_MODEL:   ${GROQ_MODEL}"
echo -e "   API URL:      ${VITE_LANGGRAPH_API_URL:-http://127.0.0.1:${LANGGRAPH_PORT}}"
if [ -n "$LANGSMITH_API_KEY" ]; then
    echo -e "   LANGSMITH_API_KEY: set"
fi

# ─── 2. Toolchain ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Step 2: Checking dependencies...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Install Python 3.11+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    echo -e "${RED}Python $PYTHON_VERSION found. Python 3.11+ is required.${NC}"
    exit 1
fi
echo -e "${GREEN}Python found: $PYTHON_VERSION${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    python3 -m pip install uv
fi
echo -e "${GREEN}uv available${NC}"

# ─── 3. Install backend + frontend ───────────────────────────────────────────
echo ""
echo -e "${YELLOW}Step 3: Syncing backend (src/)...${NC}"
uv_backend sync --quiet
echo -e "${GREEN}Backend dependencies installed${NC}"

if command -v npm &> /dev/null; then
    echo "Installing frontend dependencies..."
    (cd "${FRONTEND_DIR}" && npm install --no-audit --no-fund)
    echo -e "${GREEN}Frontend dependencies installed${NC}"
else
    echo -e "${YELLOW}npm not found — UI install skipped (need Node 20+)${NC}"
fi

# ─── 4. CLI tools ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Step 4: Verifying CLI tools...${NC}"
uv_backend run langgraph --version > /dev/null
echo -e "${GREEN}LangGraph CLI available${NC}"
uv_backend run streamlit --version > /dev/null
echo -e "${GREEN}Streamlit available${NC}"

# ─── 5. Import smoke ─────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Step 5: Verifying Python imports...${NC}"
uv_backend run python -c "
import sys
try:
    from agent import graph
    from core.config.settings import get_settings
    from core.loaders import load_document
    from core.chunking import get_chunker
    from core.vectorstore import get_vector_store
    from subagents.ocr import get_ocr_engine
    from subagents.embeddings import embed_query
    s = get_settings()
    assert graph is not None
    assert callable(load_document) and callable(get_chunker)
    assert callable(get_vector_store) and callable(get_ocr_engine)
    assert callable(embed_query)
    assert s.embedding_model_id
    print('All imports successful')
    print('Graph nodes:', sorted(n for n in graph.nodes if not n.startswith('__')))
    print('GROQ_MODEL:', s.groq_model)
    print('Models dir:', s.models_dir)
    print('OCR engine:', s.ocr_engine)
    sys.exit(0)
except Exception as e:
    print(f'Import failed: {e}')
    sys.exit(1)
" || {
    echo -e "${RED}Verification failed${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}Optional: prefetch embeddings → uv --directory src run python ../scripts/ensure_models.py${NC}"

# ─── Post actions ────────────────────────────────────────────────────────────
if [ "$SKIP_RUN" -eq 1 ]; then
    echo -e "${GREEN}Setup complete (--skip-run). Use ./start.sh both or ./start.sh prod${NC}"
    exit 0
fi

if [ -n "$POST_ACTION" ]; then
    echo ""
    echo -e "${BLUE}Starting via ./start.sh ${POST_ACTION}...${NC}"
    exec ./start.sh "$POST_ACTION"
fi

echo ""
echo -e "${BLUE}=================================================================="
echo "                     SETUP COMPLETE!"
echo -e "==================================================================${NC}"
echo ""
echo -e "${YELLOW}Local (dev — hot reload):${NC}"
echo -e "  ${GREEN}./start.sh both${NC}     → API :${LANGGRAPH_PORT} + React :${FRONTEND_PORT}"
echo -e "  ${GREEN}./start.sh ui${NC}       → React only"
echo -e "  ${GREEN}./start.sh server${NC}   → API + Studio tunnel"
echo ""
echo -e "${YELLOW}Production-like (local):${NC}"
echo -e "  ${GREEN}./start.sh prod${NC}     → build UI + API + vite preview"
echo -e "  ${GREEN}./start.sh docker${NC}   → docker compose API (src/Dockerfile)"
echo ""
echo -e "${YELLOW}Deploy:${NC}"
echo -e "  Railway Root Directory = ${GREEN}src${NC}"
echo -e "  Vercel  Root Directory = ${GREEN}frontend${NC}"
echo -e "  See docs/DEPLOYMENT.md"
echo ""
echo -e "${YELLOW}Choose an option:${NC}"
echo -e "  ${GREEN}1${NC} - Local: LangGraph + React UI  (./start.sh both)"
echo -e "  ${GREEN}2${NC} - Prod-like: build + preview    (./start.sh prod)"
echo -e "  ${GREEN}3${NC} - LangGraph only"
echo -e "  ${GREEN}4${NC} - Streamlit legacy UI"
echo -e "  ${GREEN}5${NC} - Exit"
echo ""
read -r -p "Enter your choice (1-5): " choice

case "$choice" in
    1) exec ./start.sh both ;;
    2) exec ./start.sh prod ;;
    3) exec ./start.sh server-local ;;
    4) exec ./start.sh streamlit ;;
    5) echo -e "${YELLOW}Done. Later: ./start.sh both${NC}"; exit 0 ;;
    *) echo -e "${RED}Invalid choice.${NC}"; exit 1 ;;
esac
