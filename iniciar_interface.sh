#!/usr/bin/env bash
set -euo pipefail

PASTA_PROJETO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PASTA_PROJETO"

PYTHON="python3"
if [[ -x "$PASTA_PROJETO/.venv/bin/python" ]]; then
    PYTHON="$PASTA_PROJETO/.venv/bin/python"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 não encontrado. Instale Python 3 para executar a interface."
    read -r -p "Pressione Enter para sair..."
    exit 1
fi

exec "$PYTHON" interface.py
