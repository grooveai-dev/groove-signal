# Groove Signal Service

Lightweight signaling service for the Groove decentralized inference network. Connects consumers (who need AI inference) with nodes (who provide compute) using proximity-aware scoring.

Part of the Groove network:
- [groove](https://github.com/grooveai-dev/groove) — Desktop app
- [groove-network](https://github.com/grooveai-dev/groove-network) — Node + consumer code
- **groove-signal** (this repo) — Signaling service

## What It Does

- **Node registration** — Nodes connect outbound via WebSocket, register capabilities (VRAM, RAM, GPU, bandwidth)
- **Gaussian decay scoring** — Ranks nodes by geographic proximity, uptime, compute capacity, and current load
- **Consumer matching** — Consumers query for the best available nodes to run inference
- **Envelope routing** — Routes inference traffic between matched consumers and nodes
- **Layer scheduling** — Assigns model layers to nodes proportionally based on compute capacity
- **Merkle root commitments** — Publishes signed registry snapshots for verifiable completeness

## Quick Start

### Docker

```bash
docker build -t groove-signal:v0.1.0 .
docker run -d \
  --name groove-signal \
  --restart unless-stopped \
  -p 8770:8770 \
  -p 8771:8771 \
  groove-signal:v0.1.0
```

### Manual

```bash
bash setup.sh
source venv/bin/activate
python -m src.signal.server --host 0.0.0.0 --port 8770 --http-port 8771
```

### Verify

```bash
curl http://localhost:8771/health
# {"status": "ok"}

curl http://localhost:8771/status
# {"nodes": [], "models": [], "total_nodes": 0, ...}
```

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8770 | WebSocket | Node + consumer connections |
| 8771 | HTTP | `/health` and `/status` endpoints |

## Architecture

```
Consumer ──wss──> Signal Service <──wss── Node
                      │
                      ├── Registry (node tracking + Merkle roots)
                      ├── Scorer (gaussian decay: proximity, uptime, compute, load)
                      ├── Matcher (pipeline assembly from scored nodes)
                      └── Scheduler (dynamic layer assignment)
```

Nodes connect outbound only (no open ports on user machines). The signal service never sees inference content — it brokers connections and routes envelopes.

## Scoring Algorithm

Nodes are ranked using gaussian decay across four dimensions:

| Dimension | Weight | Decay (sigma) | What It Measures |
|-----------|--------|---------------|------------------|
| Proximity | 0.35 | 500 km | Haversine distance from consumer |
| Uptime | 0.25 | 168 hrs | Hours since last registration |
| Compute | 0.20 | — | VRAM (1.0x) + RAM (0.5x) capacity |
| Load | 0.20 | 2.0 | Current system load (inverted) |

Score = weighted sum of `exp(-d^2 / 2*sigma^2)` per dimension.

## Configuration

```bash
python -m src.signal.server \
  --host 0.0.0.0 \
  --port 8770 \
  --http-port 8771 \
  --log-level INFO
```

### Optional: IP Geolocation

For proximity scoring, place a MaxMind GeoLite2-City database at `./GeoLite2-City.mmdb`. The service auto-detects it on startup. Without it, all nodes get neutral proximity scores.

## Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

58 tests covering server, scoring, registry, matcher, and scheduler.

## Production Deployment

See the [deployment guide](https://github.com/grooveai-dev/groove-signal/wiki/Deployment) or use the systemd + nginx setup:

- Systemd service for process management
- Nginx reverse proxy for TLS termination + WebSocket upgrade
- DNS A record pointing to your server

## Resource Requirements

- **CPU**: Minimal (pure networking, no ML)
- **RAM**: ~100MB base + ~1KB per registered node
- **Disk**: ~50MB (code + venv)
- **Python**: 3.10+ (3.12 recommended)

## Three-Tier Network Model

| Role | What They Do | Earns |
|------|-------------|-------|
| **Consumer** | Uses AI inference | Pays $GROOVE |
| **Node** | Lends compute, binds to localhost | Inference fees |
| **Signal** | Routes traffic + lends compute | Routing + inference fees |

For beta, Groove runs the signal service at `signal.groovedev.ai`. Post-beta, anyone can run a signal by staking $GROOVE tokens.

## License

See [LICENSE](LICENSE).
