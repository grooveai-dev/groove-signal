#!/bin/bash
# Start a Groove compute node. The node opens an outbound websocket to the
# relay and serves work received over that single connection.
#
# Usage: ./start_node.sh --model Qwen/Qwen2.5-7B --layers 0-15 --relay 1.2.3.4:8770 --device cuda

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="Qwen/Qwen2.5-7B"
LAYERS="0-15"
DEVICE="cpu"
RELAY=""
NODE_ID=""
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2 ;;
        --layers)    LAYERS="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --relay)     RELAY="$2"; shift 2 ;;
        --node-id)   NODE_ID="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --relay HOST:PORT [--model MODEL] [--layers START-END] [--device DEVICE] [--node-id ID] [--log-level LEVEL]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$RELAY" ]; then
    echo "Error: --relay HOST:PORT is required" >&2
    exit 1
fi

cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Groove compute node"
echo "  Model:  $MODEL"
echo "  Layers: $LAYERS"
echo "  Device: $DEVICE"
echo "  Relay:  $RELAY"

CMD=(python -m src.node.server
    --model "$MODEL"
    --layers "$LAYERS"
    --device "$DEVICE"
    --relay "$RELAY"
    --log-level "$LOG_LEVEL")

if [ -n "$NODE_ID" ]; then
    CMD+=(--node-id "$NODE_ID")
fi

"${CMD[@]}"
