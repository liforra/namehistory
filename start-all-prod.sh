#!/usr/bin/env bash

#!/usr/bin/env bash

# Set defaults if unset
: "${NH_API_PATH:=$(pwd)}"
: "${NH_WEB_PATH:=$(pwd)/webui}"

# Helper to source an env file if it exists
load_env() {
  if [[ -f "$1" ]]; then
    set -o allexport
    source "$1"
    set +o allexport
  fi
}

# Load from cwd
load_env "$(pwd)/.env"
load_env "$(pwd)/.env.local"
load_env "$(pwd)/webui/.env"
load_env "$(pwd)/webui/.env.local"

# Load from NH_API_PATH / NH_WEB_PATH (in case they differ from cwd)
load_env "$NH_API_PATH/.env"
load_env "$NH_API_PATH/.env.local"
load_env "$NH_WEB_PATH/.env"
load_env "$NH_WEB_PATH/.env.local"

echo "NH_API_PATH=$NH_API_PATH"
echo "NH_WEB_PATH=$NH_WEB_PATH"
if command -v uv &>/dev/null; then
  echo [API] Found uv, running it.
  uv run $NH_API_PATH/main.py
elif command -v $NH_API_PATH/.venv/bin/python3 &>/dev/null; then
  echo [API] Found Python3 inside venv at $NH_API_PATH/.venv/bin/python3, running it, might want to install uv.
  $NH_API_PATH/.venv./bin/python3 $NH_API_PATH/main.py
elif command -v python3 &>/dev/null; then
  echo [API] Found Python3 outside venv, might want to change that, running it.
  python3 $NH_API_PATH/main.py
fi
