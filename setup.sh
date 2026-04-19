#!/usr/bin/env bash
# Groove Signal setup script.
#
# Creates a venv, installs deps, verifies the install. Supports a --json
# flag for structured output consumable by the Groove daemon install flow
# (same pattern as groove-network/setup.sh).

set -e

JSON_OUTPUT=false
for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUTPUT=true ;;
    esac
done

emit() {
    local stage="$1"
    local status="$2"
    local message="$3"
    if [ "$JSON_OUTPUT" = true ]; then
        printf '{"stage":"%s","status":"%s","message":"%s"}\n' \
            "$stage" "$status" "$message"
    else
        printf '[%s] %s: %s\n' "$stage" "$status" "$message"
    fi
}

cd "$(dirname "$0")"

emit "venv" "start" "creating Python virtual environment"
if [ ! -d venv ]; then
    python3 -m venv venv
fi
emit "venv" "ok" "venv ready"

# shellcheck disable=SC1091
source venv/bin/activate

emit "pip" "start" "upgrading pip"
python -m pip install --upgrade pip --quiet
emit "pip" "ok" "pip upgraded"

emit "deps" "start" "installing requirements"
pip install --quiet -r requirements.txt
emit "deps" "ok" "requirements installed"

emit "verify" "start" "importing core modules"
python - <<'PY'
import src.common.protocol  # noqa
import src.signal.server     # noqa
import src.signal.scoring    # noqa
import src.signal.registry   # noqa
import src.signal.matcher    # noqa
import src.relay.scheduler   # noqa
PY
emit "verify" "ok" "all modules import cleanly"

emit "done" "ok" "signal service ready — run: python -m src.signal.server"
