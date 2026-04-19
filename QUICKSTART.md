# Groove Quickstart — Contribute Compute to a Relay

Groove splits a large language model across multiple machines. Each machine
runs a "node" that handles a slice of the model's layers. A central "relay"
coordinates everything — nodes connect outbound to it, so you don't need to
open ports or worry about NAT/firewalls on contributor machines.

**What you need:** Python 3.10-3.12, ~2GB disk for the 0.5B test model
(or ~15GB for the full 7B model), and network access to the relay host.

---

## 1. Install (all platforms)

### macOS

```bash
# Install Python if needed (Homebrew)
brew install python@3.11

# Extract or clone the project, then:
cd groove-deploy
bash setup.sh
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip

cd groove-deploy
bash setup.sh
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y python3.11

cd groove-deploy
bash setup.sh
```

### Windows

**Option A — WSL (recommended):**

1. Open PowerShell as admin and run: `wsl --install`
2. Restart, open Ubuntu from the Start menu
3. Follow the Linux instructions above

**Option B — Native Python:**

1. Download Python 3.11 from https://www.python.org/downloads/
   - Check "Add Python to PATH" during install
2. Open PowerShell, navigate to the project folder:

```powershell
cd groove-deploy
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU (NVIDIA CUDA) on Windows, install the CUDA version of PyTorch first:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Verify installation

```bash
source venv/bin/activate    # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

python -c "import torch; import websockets; import msgpack; print('OK')"
```

---

## 2. Architecture

```
  [Consumer]              [Relay :8770]            [Node A]  [Node B]
      |                       |                       |          |
      |--- SESSION_INIT ----->|                       |          |
      |<-- PIPELINE_CONFIG ---|                       |          |
      |                       |                       |          |
      |--- ENVELOPE(tokens)-->|--- ENVELOPE --------->|          |
      |                       |<-- ENVELOPE(hidden) --|          |
      |                       |--- ENVELOPE(hidden) ------------>|
      |<-- ENVELOPE(logits) --|<-- ENVELOPE(logits) -------------|
```

- **Relay** — the coordinator. Runs on one machine (e.g., your main computer).
  Nodes and consumers connect to it. It routes envelopes but never touches
  the model data inside them.
- **Node** — a compute worker. Loads a slice of the model and processes
  activations for its assigned layers. Connects outbound to the relay.
- **Consumer** — the inference client. Sends prompts, receives generated text.

---

## 3. Network setup with Tailscale

Groove nodes need to reach the relay. If everyone is on the same WiFi,
local IPs work. For contributors in different locations, use **Tailscale**
— a free mesh VPN that gives each machine a stable IP reachable from anywhere.

### Install Tailscale (all platforms, 2 minutes)

1. Go to https://tailscale.com/download — install for your OS
2. Sign in (Google/GitHub/etc.) — everyone joins the same Tailscale network
3. Run `tailscale ip` to get your Tailscale IP (looks like `100.x.x.x`)

That's it. Every machine on your Tailscale network can now reach every
other machine directly — no port forwarding, no firewall config, works
through any NAT.

### Which IP to use

| Scenario | Relay address |
|----------|---------------|
| Same machine | `localhost:8770` |
| Same WiFi / LAN | LAN IP (e.g., `192.168.1.50:8770`) |
| Different locations | Tailscale IP (e.g., `100.64.5.12:8770`) |

---

## 4. Start the relay

Pick one machine to be the relay (typically the person coordinating the run).

```bash
source venv/bin/activate
python -m src.relay.relay --port 8770
```

Get your IP for contributors:
```bash
tailscale ip    # Remote contributors use this
```

---

## 5. Start a compute node

Each contributor runs a node that handles a slice of the model's layers.
Coordinate who runs which layers before starting.

**Two-node example with Qwen2.5-0.5B (24 layers):**

```bash
# Machine A — first half (layers 0-11):
source venv/bin/activate
python -m src.node.server \
    --model Qwen/Qwen2.5-0.5B \
    --layers 0-11 \
    --relay 100.99.208.126:8770 \
    --device cpu

# Machine B — second half (layers 12-23):
source venv/bin/activate
python -m src.node.server \
    --model Qwen/Qwen2.5-0.5B \
    --layers 12-23 \
    --relay 100.99.208.126:8770 \
    --device cuda
```

Replace `100.99.208.126` with the relay's Tailscale IP if different
(run `tailscale ip` on the relay machine). Use `localhost` if on the
same machine.

**Solo testing (all layers on one machine):**

```bash
python -m src.node.server \
    --model Qwen/Qwen2.5-0.5B \
    --layers 0-23 \
    --relay localhost:8770 \
    --device cpu
```

**Device options:**
- `--device cpu` — works everywhere, slowest
- `--device cuda` — NVIDIA GPU (fastest)
- `--device mps` — Apple Silicon GPU (M1/M2/M3)

**The first run downloads the model** (~1GB for 0.5B, ~15GB for 7B). Subsequent
runs use the cached copy from `~/.cache/huggingface/`.

You should see:
```
[node] INFO: registered with relay node_id=node-L0-12-12345
```

### Layer splitting reference

Use `bash setup.sh --info MODEL_NAME` to see a model's layer count and
suggested splits.

| Model | Total layers | 2-node split | 3-node split |
|-------|-------------|--------------|--------------|
| Qwen/Qwen2.5-0.5B | 24 | 0-11 / 12-23 | 0-7 / 8-15 / 16-23 |
| Qwen/Qwen2.5-7B | 28 | 0-13 / 14-27 | 0-9 / 10-18 / 19-27 |

---

## 6. Run inference

Once the relay shows nodes are registered, run the consumer:

```bash
source venv/bin/activate
python -m src.consumer.client \
    --relay localhost:8770 \
    --model Qwen/Qwen2.5-0.5B \
    --prompt "The meaning of life is"
```

Options:
- `--max-tokens 200` — max tokens to generate (default 200)
- `--temperature 0.7` — sampling temperature (default 0.7)
- `--top-p 0.9` — nucleus sampling threshold (default 0.9)
- `--speculative` — enable speculative decoding (experimental)

---

## 7. Verify everything works

```bash
# Run the test suite
source venv/bin/activate
python -m pytest tests/ -v

# Run the shard smoke test (downloads Qwen2.5-0.5B)
bash setup.sh --smoke
```

---

## Troubleshooting

**"No active compute nodes available"**
- The node hasn't registered yet. Check the relay terminal for a
  `"node registered"` log line. Wait for the model to finish loading.

**"nodename nor servname provided"**
- You used a placeholder like `RELAY_IP` instead of the actual IP address.
  Use `localhost` if on the same machine, your Tailscale IP for remote
  contributors (run `tailscale ip`), or LAN IP for same-network machines.

**Node disconnects during inference**
- On CPU, large models can take a while per token. The relay has a 120s
  heartbeat timeout. If inference takes longer, the node will reconnect
  automatically.

**Connection refused from remote machine**
- If using Tailscale, make sure both machines are signed into the same
  Tailscale network and `tailscale status` shows them as connected.
- Without Tailscale, check that the relay machine's firewall allows
  inbound on port 8770:
  - macOS: System Settings > Network > Firewall > Options
  - Linux: `sudo ufw allow 8770/tcp`
  - Windows: `netsh advfirewall firewall add rule name="Groove Relay" dir=in action=allow protocol=TCP localport=8770`

**Tokens are slow on CPU**
- Expected. Qwen2.5-0.5B on CPU: ~1-3 seconds/token. Qwen2.5-7B on CPU:
  several minutes/token. Use `--device cuda` or `--device mps` for GPU
  acceleration, or use the smaller 0.5B model for testing.

**Windows: "running scripts is disabled"**
- Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell.

---

## Supported models

Any HuggingFace causal LM works in theory. Tested models:

| Model | Layers | Size | Good for |
|-------|--------|------|----------|
| Qwen/Qwen2.5-0.5B | 24 | ~1 GB | Testing, quick iteration |
| Qwen/Qwen2.5-7B | 28 | ~15 GB | Production use |

---

## File structure

```
groove-deploy/
  setup.sh                # Setup script (run first)
  requirements.txt        # Python dependencies
  QUICKSTART.md           # This file
  src/
    common/
      protocol.py         # Wire protocol v2 (msgpack + envelopes)
      tensor_transfer.py  # Tensor serialization
    node/
      server.py           # Compute node (outbound-only)
      shard_loader.py     # Model layer loading + forward pass
      kv_cache.py         # KV cache management
    consumer/
      client.py           # Inference client
      speculative.py      # Speculative decode logic
    relay/
      relay.py            # Relay router + coordinator
  tests/                  # Test suite (pytest)
  scripts/
    start_node.sh         # Node startup wrapper
    start_consumer.sh     # Consumer startup wrapper
```
