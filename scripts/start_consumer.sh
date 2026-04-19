#!/bin/bash
# Start the Groove consumer client.
#
# Usage: ./start_consumer.sh --relay localhost:8770 --model Qwen/Qwen2.5-7B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RELAY="localhost:8770"
MODEL="Qwen/Qwen2.5-7B"
PROMPT=""
MAX_TOKENS=200
SPECULATIVE=""
TEMPERATURE=0.7
TOP_P=0.9

while [[ $# -gt 0 ]]; do
    case $1 in
        --relay)       RELAY="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --prompt)      PROMPT="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --speculative) SPECULATIVE="--speculative"; shift ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p)       TOP_P="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --relay HOST:PORT --model MODEL --prompt 'TEXT' [--speculative] [--max-tokens N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$PROMPT" ]; then
    echo "Error: --prompt is required"
    exit 1
fi

cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Groove consumer client"
echo "  Relay: $RELAY"
echo "  Model: $MODEL"

python -m src.consumer.client \
    --relay "$RELAY" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    $SPECULATIVE
