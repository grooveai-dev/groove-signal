#!/bin/bash
# Groove Decentralized Inference — Setup Script
# Run this on each machine after extracting the zip.
#
# Usage:
#   bash setup.sh           # Full setup (venv + deps)
#   bash setup.sh --json    # Full setup with JSON progress output (for daemon)
#   bash setup.sh --status-json  # Output install status as single JSON object
#   bash setup.sh --test    # Run test suite after setup
#   bash setup.sh --smoke   # Run smoke test (needs Qwen2.5-0.5B download)
#   bash setup.sh --info MODEL_NAME  # Show model info + suggested layer splits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

JSON_MODE=false
LAST_PERCENT=0

log()   {
    if $JSON_MODE; then
        echo -e "${GREEN}[GROOVE]${NC} $*" >&2
    else
        echo -e "${GREEN}[GROOVE]${NC} $*"
    fi
}
warn()  {
    if $JSON_MODE; then
        echo -e "${YELLOW}[WARN]${NC} $*" >&2
    else
        echo -e "${YELLOW}[WARN]${NC} $*"
    fi
}
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# json_escape: minimally escape a string for safe inclusion in a JSON value.
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

json_emit() {
    local step="$1"
    local message="$2"
    local percent="$3"
    LAST_PERCENT="$percent"
    if $JSON_MODE; then
        printf '{"step": "%s", "message": "%s", "percent": %s}\n' \
            "$(json_escape "$step")" "$(json_escape "$message")" "$percent"
    fi
}

json_error() {
    local message="$1"
    local code="$2"
    if $JSON_MODE; then
        printf '{"step": "error", "message": "%s", "percent": %s, "code": "%s"}\n' \
            "$(json_escape "$message")" "$LAST_PERCENT" "$(json_escape "$code")"
    fi
}

json_trap_handler() {
    local exit_code=$?
    if $JSON_MODE; then
        # Emit a generic error if one wasn't already emitted by a wrapped step.
        if [[ -z "${JSON_ERROR_EMITTED:-}" ]]; then
            json_error "Setup failed (exit $exit_code)" "UNKNOWN"
        fi
    fi
    exit $exit_code
}

detect_gpu() {
    # 1. Check NVIDIA GPU — try multiple paths (Electron/daemon may have limited PATH)
    local nvidia_smi=""
    if command -v nvidia-smi &>/dev/null; then
        nvidia_smi="nvidia-smi"
    elif [[ -f "/c/Windows/System32/nvidia-smi.exe" ]]; then
        nvidia_smi="/c/Windows/System32/nvidia-smi.exe"
    elif [[ -f "/c/WINDOWS/system32/nvidia-smi.exe" ]]; then
        nvidia_smi="/c/WINDOWS/system32/nvidia-smi.exe"
    elif [[ -n "${SYSTEMROOT:-}" && -f "$SYSTEMROOT/System32/nvidia-smi.exe" ]]; then
        nvidia_smi="$SYSTEMROOT/System32/nvidia-smi.exe"
    fi

    if [[ -n "$nvidia_smi" ]] && "$nvidia_smi" &>/dev/null; then
        echo "cuda"
        return
    fi

    # 2. Check macOS (MPS for Apple Silicon, CPU for Intel)
    if [[ "$(uname -s)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo "mps"
        else
            echo "macos-cpu"
        fi
        return
    fi

    echo "cpu"
}

find_python() {
    if python3 --version &>/dev/null; then
        echo "python3"
    elif python --version &>/dev/null; then
        echo "python"
    else
        echo ""
    fi
}

activate_venv() {
    if [[ -f "venv/Scripts/activate" ]]; then
        # shellcheck disable=SC1091
        source venv/Scripts/activate
    elif [[ -f "venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
}

PYTHON_CMD="$(find_python)"
DETECTED_PY_VERSION=""

check_python_version() {
    json_emit "checking-python" "Checking Python version..." 5
    if [[ -z "$PYTHON_CMD" ]]; then
        error "python3 not found. Please install Python 3.10–3.12."
        json_error "python3 not found" "PYTHON_VERSION"
        JSON_ERROR_EMITTED=1
        exit 1
    fi
    local py_version
    py_version=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null) || true
    if [[ -z "$py_version" ]]; then
        error "Python found ($PYTHON_CMD) but failed to run. Check your Python installation."
        json_error "python3 not working" "PYTHON_VERSION"
        JSON_ERROR_EMITTED=1
        exit 1
    fi
    local major minor
    major=$(echo "$py_version" | cut -d. -f1)
    minor=$(echo "$py_version" | cut -d. -f2)
    if (( major < 3 || (major == 3 && minor < 10) )); then
        error "Python $py_version is too old. Groove requires Python 3.10 or newer."
        json_error "Python $py_version too old (need 3.10+)" "PYTHON_VERSION"
        JSON_ERROR_EMITTED=1
        exit 1
    fi
    if (( major == 3 && minor >= 13 )); then
        warn "Python $py_version detected. PyTorch may not fully support Python 3.13+."
        local found_compat=false
        for alt in python3.12 python3.11 python3.10; do
            if command -v "$alt" &>/dev/null; then
                local alt_ver
                alt_ver=$("$alt" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
                if [[ -n "$alt_ver" ]]; then
                    log "Found compatible $alt ($alt_ver) on PATH — using it instead."
                    PYTHON_CMD="$alt"
                    py_version="$alt_ver"
                    found_compat=true
                    break
                fi
            fi
        done
        if [[ "$found_compat" == false ]] && command -v py &>/dev/null; then
            for pyver in 3.12 3.11 3.10; do
                local alt_ver
                alt_ver=$(py -"$pyver" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
                if [[ -n "$alt_ver" ]]; then
                    local py_path
                    py_path=$(py -"$pyver" -c "import sys; print(sys.executable)" 2>/dev/null)
                    if [[ -n "$py_path" ]]; then
                        log "Found compatible Python $alt_ver via py launcher ($py_path)"
                        PYTHON_CMD="$py_path"
                        py_version="$alt_ver"
                        found_compat=true
                        break
                    fi
                fi
            done
        fi
        if [[ "$found_compat" == false ]]; then
            error "No Python 3.10–3.12 found. PyTorch requires Python 3.12 or older."
            error "Install Python 3.12: https://www.python.org/downloads/"
            json_error "Python $py_version not supported by PyTorch (need 3.10-3.12)" "PYTHON_VERSION"
            JSON_ERROR_EMITTED=1
            exit 1
        fi
    fi
    DETECTED_PY_VERSION="$py_version"
    log "Python version: $py_version (using $PYTHON_CMD)"
    json_emit "checking-python" "Python $py_version found" 10
}

install_deps() {
    if $JSON_MODE; then
        trap json_trap_handler EXIT
    fi

    check_python_version

    json_emit "creating-venv" "Creating virtual environment..." 15
    if [[ -d "venv" ]]; then
        activate_venv
        local venv_py_ver=""
        venv_py_ver=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
        local venv_minor="${venv_py_ver#3.}"
        if [[ -n "$venv_py_ver" ]] && (( ${venv_minor:-99} >= 10 && ${venv_minor:-99} <= 12 )); then
            log "Reusing existing venv (Python $venv_py_ver)"
            json_emit "creating-venv" "Existing venv OK (Python $venv_py_ver)" 25
        else
            log "Existing venv has Python $venv_py_ver — recreating with $PYTHON_CMD..."
            rm -rf venv
            if ! $PYTHON_CMD -m venv venv 2>&1; then
                error "Failed to create virtual environment"
                json_error "Failed to create virtual environment" "VENV_CREATE"
                JSON_ERROR_EMITTED=1
                exit 1
            fi
            activate_venv
            json_emit "creating-venv" "Virtual environment recreated" 25
        fi
    else
        log "Creating Python virtual environment..."
        if ! $PYTHON_CMD -m venv venv 2>&1; then
            error "Failed to create virtual environment"
            json_error "Failed to create virtual environment" "VENV_CREATE"
            JSON_ERROR_EMITTED=1
            exit 1
        fi
        activate_venv
        json_emit "creating-venv" "Virtual environment created" 25
    fi

    log "Upgrading pip..."
    "$PYTHON_CMD" -m pip install --upgrade pip --quiet >&2 2>/dev/null || "$PYTHON_CMD" -m pip install --upgrade pip --quiet

    local gpu_type
    gpu_type=$(detect_gpu)
    log "Detected compute device: $gpu_type"

    # Install PyTorch BEFORE requirements.txt so accelerate doesn't
    # pull in the CPU-only build from PyPI.
    json_emit "installing-torch" "Installing PyTorch ($gpu_type)..." 30

    local existing_device=""
    existing_device=$("$PYTHON_CMD" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "")

    local need_install=true
    if [[ -n "$existing_device" ]]; then
        if [[ "$gpu_type" == "cuda" && "$existing_device" == "cuda" ]]; then
            need_install=false
        elif [[ "$gpu_type" == "mps" && "$existing_device" == "mps" ]]; then
            need_install=false
        elif [[ "$gpu_type" == "macos-cpu" && -n "$existing_device" ]]; then
            need_install=false
        elif [[ "$gpu_type" == "cpu" && -n "$existing_device" ]]; then
            need_install=false
        fi
    fi

    install_torch() {
        local torch_ok=0
        if [[ "$gpu_type" == "cuda" ]]; then
            log "Installing PyTorch with CUDA support (this may take a few minutes)..."
            "$PYTHON_CMD" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --timeout 300 >&2 && torch_ok=1
            if (( torch_ok == 0 )); then
                log "cu124 unavailable, trying cu121..."
                "$PYTHON_CMD" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --timeout 300 >&2 && torch_ok=1
            fi
        elif [[ "$gpu_type" == "mps" ]]; then
            log "Installing PyTorch with MPS (Apple Silicon) support..."
            "$PYTHON_CMD" -m pip install torch >&2 && torch_ok=1
        elif [[ "$gpu_type" == "macos-cpu" ]]; then
            log "Installing PyTorch for macOS Intel (CPU)..."
            "$PYTHON_CMD" -m pip install torch >&2 && torch_ok=1
        else
            log "Installing PyTorch (Linux CPU only)..."
            "$PYTHON_CMD" -m pip install torch --index-url https://download.pytorch.org/whl/cpu >&2 && torch_ok=1
        fi
        return $(( 1 - torch_ok ))
    }

    if $need_install; then
        if ! install_torch; then
            error "Failed to install PyTorch"
            json_error "Failed to install PyTorch" "TORCH_INSTALL"
            JSON_ERROR_EMITTED=1
            exit 1
        fi
    else
        log "PyTorch already installed with $existing_device support — skipping"
    fi
    json_emit "installing-torch" "PyTorch installed" 50

    json_emit "installing-deps" "Installing dependencies..." 55
    log "Installing remaining dependencies..."
    if ! "$PYTHON_CMD" -m pip install -r requirements.txt --quiet; then
        error "Failed to install dependencies from requirements.txt"
        json_error "Failed to install dependencies" "DEPS_INSTALL"
        JSON_ERROR_EMITTED=1
        exit 1
    fi
    json_emit "installing-deps" "Dependencies installed" 70

    json_emit "verifying" "Verifying installation..." 80
    log "Verifying installation..."

    # Ask Python what actually works — this is the source of truth.
    local actual_device
    actual_device=$("$PYTHON_CMD" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu")

    # Auto-correct: GPU detected by system but PyTorch can't use it.
    # Try to replace CPU torch with CUDA build. If it fails, ensure
    # CPU torch still works so the node can at least run.
    if [[ "$gpu_type" == "cuda" && "$actual_device" != "cuda" ]]; then
        warn "CUDA GPU detected but PyTorch reports CPU-only"
        warn "Attempting to install CUDA PyTorch..."
        json_emit "installing-torch" "Reinstalling PyTorch with CUDA (fixing CPU-only build)..." 82
        "$PYTHON_CMD" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --timeout 300 >&2 && \
            actual_device=$("$PYTHON_CMD" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
else:
    print('cpu')
" 2>/dev/null || echo "cpu")
        if [[ "$actual_device" != "cuda" ]]; then
            # CUDA install failed — make sure CPU torch is still usable
            "$PYTHON_CMD" -c "import torch" 2>/dev/null || \
                "$PYTHON_CMD" -m pip install torch --timeout 300 >&2
            warn "Could not enable CUDA — node will run on CPU."
            warn "To fix manually: pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall"
        fi
    fi

    local verify_output
    if ! verify_output=$("$PYTHON_CMD" -c "
import torch
import transformers
import websockets
import msgpack
import numpy
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  websockets:   {websockets.__version__}')
print(f'  msgpack:      {msgpack.version}')
print(f'  numpy:        {numpy.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
print(f'  MPS:          {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
" 2>&1); then
        error "Verification failed:"
        error "$verify_output"
        json_error "Package verification failed" "VERIFY_FAILED"
        JSON_ERROR_EMITTED=1
        exit 1
    fi
    if $JSON_MODE; then
        echo "$verify_output" >&2
    else
        echo "$verify_output"
    fi
    json_emit "verifying" "All packages verified" 95

    local torch_device="$actual_device"

    local torch_version
    torch_version=$("$PYTHON_CMD" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

    # Emit the done event with extra fields (device, python, torch).
    if $JSON_MODE; then
        printf '{"step": "done", "message": "%s", "percent": 100, "device": "%s", "python": "%s", "torch": "%s"}\n' \
            "$(json_escape "Setup complete")" \
            "$(json_escape "$torch_device")" \
            "$(json_escape "$DETECTED_PY_VERSION")" \
            "$(json_escape "$torch_version")"
        LAST_PERCENT=100
        # Clear the trap so EXIT handler doesn't emit a spurious error.
        trap - EXIT
        return 0
    fi

    local local_ip
    local_ip=$($PYTHON_CMD -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
except Exception:
    print('localhost')
finally:
    s.close()
" 2>/dev/null || echo "localhost")

    local tailscale_ip
    tailscale_ip=$(tailscale ip -4 2>/dev/null || echo "")

    log "Setup complete! Device: $gpu_type"
    echo ""
    echo "========================================="
    echo "  Groove is ready on this machine."
    echo "  Device: $gpu_type"
    echo "  LAN IP: $local_ip"
    if [[ -n "$tailscale_ip" ]]; then
        echo "  Tailscale IP: $tailscale_ip"
    fi
    echo "========================================="
    echo ""
    echo "Next steps:"
    if [[ -f "venv/Scripts/activate" ]]; then
        echo "  source venv/Scripts/activate"
    else
        echo "  source venv/bin/activate"
    fi
    echo ""
    echo "  # 1. Start the relay (coordinator machine):"
    echo "  python -m src.relay.relay --port 8770"
    echo ""

    if [[ -n "$tailscale_ip" ]]; then
        echo "  # 2. Start a compute node:"
        echo "  #    Same machine as relay:"
        echo "  python -m src.node.server --model Qwen/Qwen2.5-0.5B --layers 0-23 --relay localhost:8770 --device $torch_device"
        echo "  #    Remote machine (via Tailscale):"
        echo "  python -m src.node.server --model Qwen/Qwen2.5-0.5B --layers 0-23 --relay $tailscale_ip:8770 --device $torch_device"
    else
        echo "  # 2. Start a compute node:"
        echo "  #    Same machine as relay:"
        echo "  python -m src.node.server --model Qwen/Qwen2.5-0.5B --layers 0-23 --relay localhost:8770 --device $torch_device"
        echo "  #    Remote machine (LAN):"
        echo "  python -m src.node.server --model Qwen/Qwen2.5-0.5B --layers 0-23 --relay $local_ip:8770 --device $torch_device"
        echo ""
        echo "  For remote contributors, install Tailscale: https://tailscale.com/download"
    fi
    echo ""
    echo "  # 3. Run inference (consumer):"
    echo "  python -m src.consumer.client --relay localhost:8770 --model Qwen/Qwen2.5-0.5B --prompt 'Hello world'"
    echo ""
    echo "  # Run tests:"
    echo "  python -m pytest tests/ -v"
    echo ""
    echo "  # Quick smoke test:"
    echo "  bash setup.sh --smoke"
}

run_tests() {
    if [[ ! -d "venv" ]]; then
        error "Run 'bash setup.sh' first to install dependencies"
        exit 1
    fi
    activate_venv
    log "Running test suite..."
    python -m pytest tests/ -v
}

run_smoke() {
    if [[ ! -d "venv" ]]; then
        error "Run 'bash setup.sh' first to install dependencies"
        exit 1
    fi
    activate_venv

    local gpu_type
    gpu_type=$(detect_gpu)
    log "Running smoke test with Qwen2.5-0.5B on $gpu_type..."

    $PYTHON_CMD -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.node.shard_loader import forward_shard
import torch.nn as nn

MODEL = 'Qwen/Qwen2.5-0.5B'
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

text = 'The quick brown fox jumps over the lazy dog'
inputs = tokenizer(text, return_tensors='pt')

print('Running full model...')
with torch.no_grad():
    full_out = model(**inputs)
    full_logits = full_out.logits

num_layers = len(model.model.layers)
mid = num_layers // 2
print(f'Splitting into 2 shards: [0, {mid}) and [{mid}, {num_layers})')

def make_shard(m, start, end):
    total = len(m.model.layers)
    return {
        'layers': nn.ModuleList([m.model.layers[i] for i in range(start, end)]),
        'embed_tokens': m.model.embed_tokens if start == 0 else None,
        'norm': m.model.norm if end == total else None,
        'lm_head': m.lm_head if end == total else None,
        'rotary_emb': m.model.rotary_emb if hasattr(m.model, 'rotary_emb') else None,
        'config': m.config,
        'layer_start': start,
        'layer_end': end,
        'total_layers': total,
    }

shard1 = make_shard(model, 0, mid)
shard2 = make_shard(model, mid, num_layers)

print('Running through shards...')
with torch.no_grad():
    hidden, _ = forward_shard(shard1, inputs['input_ids'])
    shard_logits, _ = forward_shard(shard2, hidden)

max_diff = (full_logits - shard_logits).abs().max().item()
full_next = torch.argmax(full_logits[:, -1, :], dim=-1).item()
shard_next = torch.argmax(shard_logits[:, -1, :], dim=-1).item()

print(f'Max logit difference: {max_diff:.6f}')
print(f'Full model next token:  {full_next} ({tokenizer.decode([full_next])})')
print(f'Shard model next token: {shard_next} ({tokenizer.decode([shard_next])})')

if max_diff < 1e-3 and full_next == shard_next:
    print('SMOKE TEST PASSED - shards produce identical output')
else:
    print('SMOKE TEST FAILED')
    exit(1)
"
}

show_info() {
    local model_name="${1:-Qwen/Qwen2.5-0.5B}"
    if [[ ! -d "venv" ]]; then
        error "Run 'bash setup.sh' first to install dependencies"
        exit 1
    fi
    activate_venv

    $PYTHON_CMD - "$model_name" <<'PYEOF'
import sys
from src.node.shard_loader import get_model_info
model_name = sys.argv[1]
info = get_model_info(model_name)
print(f'Model: {model_name}')
print(f'  Total layers:  {info["total_layers"]}')
print(f'  Hidden size:   {info["hidden_size"]}')
print(f'  Attention heads: {info["num_heads"]}')
print(f'  KV heads:      {info["num_kv_heads"]}')
print(f'  Vocab size:    {info["vocab_size"]}')
print(f'  Model type:    {info["model_type"]}')
print(f'  Dtype:         {info["dtype"]}')
print()
n = info['total_layers']
print('Suggested layer splits:')
print(f'  2 nodes: [0-{n//2 - 1}] [{n//2}-{n-1}]')
t = n // 3
print(f'  3 nodes: [0-{t-1}] [{t}-{2*t-1}] [{2*t}-{n-1}]')
PYEOF
}

show_status() {
    echo "=== Groove Deploy Status ==="
    echo "Directory: $SCRIPT_DIR"
    echo ""

    if [[ -d "venv" ]]; then
        echo "Venv: INSTALLED"
        activate_venv
        $PYTHON_CMD -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
print(f'  MPS:     {mps}')
" 2>/dev/null || echo "  (could not import torch)"
    else
        echo "Venv: NOT INSTALLED — run 'bash setup.sh'"
    fi

    echo ""
    echo "Source files:"
    for f in src/common/protocol.py src/node/server.py src/relay/relay.py src/consumer/client.py; do
        if [[ -f "$f" ]]; then
            echo "  $f: OK"
        else
            echo "  $f: MISSING"
        fi
    done
}

show_status_json() {
    # Output a single JSON object describing current install status.
    # Consumed by the daemon's GET /api/network/install/status endpoint.
    local installed=false
    local venv=false
    local py_version=""
    local torch_version=""
    local device=""
    local cuda=false
    local mps=false

    if [[ -d "venv" ]]; then
        venv=true
        if [[ -f "venv/Scripts/activate" ]] || [[ -f "venv/bin/activate" ]]; then
            # shellcheck disable=SC1091
            activate_venv
            py_version=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null || echo "")
            local probe
            probe=$($PYTHON_CMD -c "
import json
try:
    import torch
    cuda = bool(torch.cuda.is_available())
    mps = bool(torch.backends.mps.is_available()) if hasattr(torch.backends, 'mps') else False
    if cuda:
        device = 'cuda'
    elif mps:
        device = 'mps'
    else:
        device = 'cpu'
    print(json.dumps({'torch': torch.__version__, 'cuda': cuda, 'mps': mps, 'device': device}))
except Exception:
    print(json.dumps({'torch': '', 'cuda': False, 'mps': False, 'device': ''}))
" 2>/dev/null || echo '{"torch": "", "cuda": false, "mps": false, "device": ""}')
            torch_version=$($PYTHON_CMD -c "import json,sys; d=json.loads('''$probe'''); print(d.get('torch',''))" 2>/dev/null || echo "")
            cuda=$($PYTHON_CMD -c "import json; d=json.loads('''$probe'''); print('true' if d.get('cuda') else 'false')" 2>/dev/null || echo "false")
            mps=$($PYTHON_CMD -c "import json; d=json.loads('''$probe'''); print('true' if d.get('mps') else 'false')" 2>/dev/null || echo "false")
            device=$($PYTHON_CMD -c "import json; d=json.loads('''$probe'''); print(d.get('device',''))" 2>/dev/null || echo "")
            if [[ -n "$torch_version" ]]; then
                installed=true
            fi
        fi
    fi

    printf '{"installed": %s, "python": "%s", "torch": "%s", "device": "%s", "cuda": %s, "mps": %s, "venv": %s}\n' \
        "$installed" \
        "$(json_escape "$py_version")" \
        "$(json_escape "$torch_version")" \
        "$(json_escape "$device")" \
        "$cuda" \
        "$mps" \
        "$venv"
}

case "${1:-}" in
    --test)
        run_tests
        ;;
    --smoke)
        run_smoke
        ;;
    --info)
        show_info "${2:-Qwen/Qwen2.5-7B}"
        ;;
    --status)
        show_status
        ;;
    --status-json)
        show_status_json
        ;;
    --json)
        JSON_MODE=true
        install_deps
        ;;
    --help|-h)
        echo "Groove Decentralized Inference — Setup"
        echo ""
        echo "Usage:"
        echo "  bash setup.sh              Install dependencies and set up venv"
        echo "  bash setup.sh --json       Install with JSON progress output (for daemon)"
        echo "  bash setup.sh --status     Check installation health"
        echo "  bash setup.sh --status-json  Install status as JSON (for daemon)"
        echo "  bash setup.sh --test       Run the test suite"
        echo "  bash setup.sh --smoke      Run smoke test (Qwen2.5-0.5B)"
        echo "  bash setup.sh --info MODEL Show model info + layer splits"
        echo "  bash setup.sh --help       Show this help"
        ;;
    *)
        install_deps
        ;;
esac
