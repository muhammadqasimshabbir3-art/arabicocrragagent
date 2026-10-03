#!/bin/bash
# Shared helpers for starting/stopping Arabic Document Intelligence services.

LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

load_dotenv() {
    local env_file="${1:-.env}"
    if [ ! -f "$env_file" ]; then
        return 1
    fi

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
        esac

        local key="${line%%=*}"
        local value="${line#*=}"

        key="$(echo "$key" | xargs)"
        if [ -z "$key" ]; then
            continue
        fi

        if [[ "$value" =~ ^\"(.*)\"$ ]]; then
            value="${BASH_REMATCH[1]}"
        elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi

        export "${key}=${value}"
    done < "$env_file"
}

# Ensure local .env has values needed for frontend ↔ backend connectivity.
ensure_local_env_defaults() {
    local env_file="${1:-.env}"
    [ -f "$env_file" ] || return 1

    if ! grep -q '^VITE_LANGGRAPH_API_URL=' "$env_file"; then
        echo "VITE_LANGGRAPH_API_URL=http://127.0.0.1:${LANGGRAPH_PORT}" >> "$env_file"
    fi
    if ! grep -q '^VITE_LANGGRAPH_ASSISTANT_ID=' "$env_file"; then
        echo "VITE_LANGGRAPH_ASSISTANT_ID=agent" >> "$env_file"
    fi
    if ! grep -q '^GROQ_MODEL=' "$env_file"; then
        echo "GROQ_MODEL=openai/gpt-oss-120b" >> "$env_file"
    fi
    if ! grep -q '^OCR_ENGINE=' "$env_file"; then
        echo "OCR_ENGINE=digital" >> "$env_file"
    fi
    if ! grep -q '^CORS_ALLOW_ORIGINS=' "$env_file"; then
        echo "CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501,http://127.0.0.1:8501,http://localhost:2024,http://127.0.0.1:2024,https://smith.langchain.com" >> "$env_file"
    else
        # Append missing local UI origins without rewriting production CORS.
        local cors
        cors="$(grep '^CORS_ALLOW_ORIGINS=' "$env_file" | head -1 | cut -d= -f2-)"
        local need origin
        for origin in \
            "http://localhost:${FRONTEND_PORT}" \
            "http://127.0.0.1:${FRONTEND_PORT}" \
            "http://localhost:${LANGGRAPH_PORT}" \
            "http://127.0.0.1:${LANGGRAPH_PORT}"; do
            case ",${cors}," in
                *",${origin},"*) ;;
                *) cors="${cors},${origin}" ;;
            esac
        done
        # Normalize accidental leading comma
        cors="${cors#,}"
        sed -i "s|^CORS_ALLOW_ORIGINS=.*|CORS_ALLOW_ORIGINS=${cors}|" "$env_file"
    fi
}

link_backend_env() {
    local repo_root="${1:-.}"
    local backend_dir="${2:-src}"
    ln -sfn ../.env "${backend_dir}/.env" 2>/dev/null || cp -f "${repo_root}/.env" "${backend_dir}/.env"
}

stop_port() {
    local port="$1"
    local pids=""

    if command -v lsof &>/dev/null; then
        pids="$(lsof -ti:"${port}" 2>/dev/null || true)"
    elif command -v fuser &>/dev/null; then
        fuser -k "${port}/tcp" 2>/dev/null || true
        sleep 1
        return 0
    elif command -v ss &>/dev/null; then
        pids="$(ss -lptn "sport = :${port}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\n' ' ')"
    fi

    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill ${pids} 2>/dev/null || true
        sleep 1
        # shellcheck disable=SC2086
        kill -9 ${pids} 2>/dev/null || true
    fi

    sleep 1
}

stop_langgraph() {
    pkill -f "langgraph dev" 2>/dev/null || true
    stop_port "${LANGGRAPH_PORT}"
}

stop_streamlit() {
    pkill -f "streamlit run streamlit_ui.py" 2>/dev/null || true
    stop_port "${STREAMLIT_PORT}"
}

stop_frontend() {
    pkill -f "vite" 2>/dev/null || true
    pkill -f "vite preview" 2>/dev/null || true
    stop_port "${FRONTEND_PORT}"
}

stop_all_services() {
    echo "Stopping services on ports ${LANGGRAPH_PORT}, ${STREAMLIT_PORT}, and ${FRONTEND_PORT}..."
    stop_langgraph
    stop_streamlit
    stop_frontend
    echo "Services stopped."
}

wait_for_port() {
    local port="$1"
    local retries="${2:-45}"
    local i=0

    while [ "$i" -lt "$retries" ]; do
        if command -v curl &>/dev/null && curl -fsS --max-time 1 "http://127.0.0.1:${port}/ok" >/dev/null 2>&1; then
            return 0
        fi
        if command -v curl &>/dev/null && curl -fsS --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done

    echo "Warning: port ${port} not ready after ${retries}s (continuing anyway)" >&2
    return 0
}
