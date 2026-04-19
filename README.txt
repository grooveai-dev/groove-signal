GROOVE DECENTRALIZED INFERENCE — DEPLOY PACKAGE
================================================

Quick Start
-----------
1. Extract this folder on each machine
2. Run: bash setup.sh
3. Follow the printed instructions

See QUICKSTART.md for detailed setup guides (macOS, Linux, Windows).

Commands
--------
SETUP (run once per machine):
  bash setup.sh              # Install Python venv + all dependencies
  bash setup.sh --status     # Check installation health
  bash setup.sh --test       # Run test suite
  bash setup.sh --smoke      # Smoke test with Qwen2.5-0.5B
  bash setup.sh --info MODEL # Show model info + suggested layer splits

RELAY (run on coordinator machine):
  source venv/bin/activate
  python -m src.relay.relay --port 8770

COMPUTE NODE (run on each contributor machine):
  source venv/bin/activate
  python -m src.node.server \
    --model Qwen/Qwen2.5-0.5B \
    --layers 0-23 \
    --relay RELAY_IP:8770 \
    --device cpu

  Replace RELAY_IP with the relay's actual IP (or localhost if same machine).
  Use --device cuda for NVIDIA GPU, --device mps for Apple Silicon.

  Layer ranges are inclusive. Split examples for Qwen2.5-0.5B (24 layers):
    Machine A:  --layers 0-11
    Machine B:  --layers 12-23

CONSUMER (run inference):
  source venv/bin/activate
  python -m src.consumer.client \
    --relay localhost:8770 \
    --model Qwen/Qwen2.5-0.5B \
    --prompt "Hello world"

Network Setup
-------------
- The relay binds to 0.0.0.0:8770 — accessible from the LAN by default
- Nodes connect OUTBOUND to the relay — no port forwarding needed on
  contributor machines
- Only the relay machine needs its firewall port open (8770/tcp)
- The consumer and nodes never communicate directly; all traffic flows
  through the relay via envelope routing (protocol v2)

File Structure
--------------
groove-deploy/
  setup.sh              # Setup script (run first)
  requirements.txt      # Python dependencies
  QUICKSTART.md         # Detailed setup guide (Mac/Linux/Windows)
  README.txt            # This file
  src/
    common/
      protocol.py       # Wire protocol v2 (msgpack + envelopes)
      tensor_transfer.py # Tensor serialization
    node/
      server.py         # Compute node (outbound-only)
      shard_loader.py   # Model layer loading + forward pass
      kv_cache.py       # KV cache management
    consumer/
      client.py         # Inference client
      speculative.py    # Speculative decode logic
    relay/
      relay.py          # Relay router + coordinator
  tests/                # Test suite (pytest)
  scripts/
    start_node.sh       # Node startup wrapper
    start_consumer.sh   # Consumer startup wrapper
