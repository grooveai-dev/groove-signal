# Groove Roadmap

## Vision

A decentralized, crypto-incentivized global LLM inference network — PiperNet for AI.

Anyone with a gaming rig, Mac Mini, or spare GPU contributes compute and earns
$GROOVE tokens. Anyone who needs intelligence buys it through the network. No
massive data centers, no single point of failure, no gatekeepers. Like Bitcoin
mining rewarded Proof of Work, Groove rewards Proof of Compute (PoC).

The Groove desktop app is the entry point. Open it, and you're either lending
compute power to the network or consuming it — seamlessly routed through smart
contracts on Base L2. Over time, the network trains open-source Savant LLMs on
anonymized usage patterns, creating domain-specialist models that get smarter
with every query. A farmer in rural India gets PhD-level medical advice for
fractions of a cent. A student in Lagos gets a code tutor as good as any
Silicon Valley engineer. Intelligence as a public utility.

**Tech stack:** Python (asyncio + websockets + PyTorch), Base L2 (EVM/Solidity),
Groove desktop app (Electron/Tauri — separate repo at ~/Desktop/groove).

---

## M1 — Relay-Mediated Data Plane (COMPLETE)

The foundation. Nodes connect outbound to a relay, all inference traffic
flows through opaque envelopes, consumers never learn node addresses.

**Delivered:**
- Protocol v2: envelope multiplexing with `stream_id` + `target_node_id`
- Node lifecycle: `REGISTER_NODE` / `REGISTER_ACK` / `DEREGISTER` / heartbeats
- Relay: pure router with JSON structured logging, no payload inspection
- NAT-friendly: nodes connect outbound, no port forwarding needed
- Multi-node pipeline: consumer walks nodes in layer order via relay
- Graceful shutdown: SIGINT sends DEREGISTER, reconnect with backoff
- Thread-offloaded inference: heartbeats survive long CPU forward passes
- Speculative decoding: local draft model + relay verification (experimental)

**Architecture:**
```
Consumer ---ws---> Relay:8770 <---ws--- Node A (layers 0-11)
                              <---ws--- Node B (layers 12-23)
```

**Codebase (as of M1):**
```
src/common/protocol.py       — message types, msgpack encoding
src/common/tensor_transfer.py — binary tensor serialization
src/relay/relay.py            — relay router, heartbeat monitor
src/node/server.py            — compute node, inference dispatch
src/node/shard_loader.py      — HuggingFace model shard loading
src/node/kv_cache.py          — per-session KV cache management
src/consumer/client.py        — inference client, pipeline walker
src/consumer/speculative.py   — speculative decoding
src/consumer/draft_model.py   — local draft model for speculation
```

---

## M2 — Dynamic Layer Assignment (COMPLETE)

Removed manual `--layers` coordination. Nodes register with capabilities;
the relay assigns layer ranges based on capacity and current coverage.

**Delivered:**
- Nodes register with capabilities only (RAM, VRAM, bandwidth, device, CPU
  cores, GPU model, max context length) — no `--layers` flag required
- Relay maintains a model registry (layer count, hidden size, memory per layer)
  with Qwen2.5-0.5B as first entry
- Standalone scheduler module assigns layer slices proportional to node capacity:
  GPU nodes weighted by VRAM, CPU nodes at 0.5x RAM effective capacity
- On-demand shard loading: nodes fetch only their assigned layers from
  HuggingFace after receiving ASSIGN_LAYERS from relay
- Rebalancing: when nodes join/leave, relay recalculates and sends REBALANCE
  to affected nodes (new sessions only — active sessions untouched)
- Layer coverage validation before accepting consumer sessions
- Envelope counter (uint64) added to wire format — M4 metering prep
- Node identity: secp256k1 keypairs with Ethereum-style addresses (0x...),
  persisted to ~/.groove/node_key.json — maps directly to Base L2 wallets in M4
- Legacy --layers override preserved for development/testing
- 48 new test cases covering scheduler, identity, dynamic assignment, integration

**New protocol messages:**
- `ASSIGN_LAYERS` — relay -> node, specifying model + layer range + metadata
- `ASSIGNMENT_ACK` — node -> relay, confirming shard loaded (with load_time_ms)
  or rejected (with reason)
- `REBALANCE` — relay -> node, requesting layer range change

**New files:**
```
src/relay/scheduler.py   — portable scheduler (no async/ws deps, ready for M3)
src/node/identity.py     — secp256k1 keypair gen, Ethereum address derivation
```

**Key design decisions:**
- Scheduler is a pure-logic module with zero relay dependencies — it will
  move into the consensus layer in M3 without modification
- Node IDs are now Ethereum addresses, not random strings — direct wallet
  mapping for M4 token payments on Base L2
- Envelope counter lays groundwork for per-envelope billing without
  another protocol revision
- Rebalancing is non-preemptive: active sessions finish on old assignments,
  new sessions get the optimized layout

**Known limitations (acceptable for M2, addressed later):**
- Single model in registry (Qwen2.5-0.5B) — multi-model scheduling is M3
- No rollback if partial rebalance fails (some nodes reject) — M6 hardening
- 120s assignment timeout generous but untuned — M6 hardening
- Only Llama/Qwen2.5 architectures supported in shard loader

---

## M3 — Decentralized Relay (DHT + Gossip)

Eliminate the single relay as a point of failure. Every node embeds relay
logic and participates in a peer-to-peer coordination layer.

**Goals:**
- Each node runs both compute and relay logic
- Nodes discover each other via DHT (distributed hash table),
  similar to BitTorrent/IPFS
- Bootstrap nodes: small set of well-known entry points for initial
  peer discovery (hardcoded or DNS-seeded)
- Session routing: any node can be a session entry point; it discovers
  which peers hold the needed layers and assembles a pipeline
- Gossip protocol: nodes share layer assignments, capabilities,
  and health status with neighbors
- Partition tolerance: if the network splits, each partition can still
  serve models if it has full layer coverage
- Multi-model registry distributed via gossip (not hardcoded)

**Architecture evolution:**
```
        [Node A] <--gossip--> [Node B]
         /    \                /    \
    [Node C]  [Node D]   [Node E]  [Node F]

Consumer connects to ANY node -> that node becomes the session relay
```

**Implementation steps:**

Step 1 — Embedded relay mode:
- Merge relay logic into node process (node can serve as both compute + router)
- Single binary that can act as relay-only, compute-only, or both
- Config flag: --mode relay | compute | full
- This alone eliminates the separate relay deployment

Step 2 — Peer discovery (DHT):
- Implement Kademlia-style DHT for node discovery
- Bootstrap node list (hardcoded IPs or DNS TXT records)
- Nodes announce themselves to DHT on startup
- DHT stores: node_id (ETH address) -> (capabilities, layers, endpoint)
- Consider using existing library (e.g., kademlia Python package)

Step 3 — Gossip protocol:
- Periodic state exchange between connected peers
- Each node maintains a partial view of the network
- Gossip messages: node join/leave, layer assignments, health updates
- Anti-entropy: periodically reconcile state with random peers
- Crytographic node identity (from M2) prevents spoofing

Step 4 — Decentralized scheduling:
- Move scheduler.py into consensus layer
- Leader election per model (lightweight, e.g., highest-staked node)
- Leader runs scheduler, broadcasts assignments via gossip
- Fallback: if no leader, nodes self-assign based on local view
- Consistency: eventual consistency is acceptable (not strict)

Step 5 — Session assembly without central relay:
- Consumer connects to any node (entry node)
- Entry node queries DHT/gossip for nodes covering needed layers
- Entry node assembles pipeline and routes envelopes
- If entry node goes down, consumer reconnects to another node

**Key challenges:**
- Consistency without central authority — leader election per model
- Gossip propagation latency vs. central relay speed
- Sybil resistance — ties into M4 staking (fake nodes must lock tokens)
- NAT traversal for peer-to-peer: STUN/TURN or relay-assisted hole punching

**Dependencies:** M2 (scheduler portability, node identity)

---

## M4 — Token Economics + Incentive Layer ($GROOVE on Base L2)

Crypto-economic layer that rewards compute contributors and charges
consumers. The $GROOVE token creates a self-sustaining marketplace
for inference on the Base L2 network (Coinbase's Ethereum L2).

**Why Base L2:**
- EVM-compatible (Solidity smart contracts)
- Low gas fees (fractions of a cent per tx)
- Fast finality (~2 seconds)
- Strong ecosystem (Coinbase, large user base)
- Team has existing Ethereum/blockchain development experience

**Goals:**

Token contract:
- ERC-20 $GROOVE token on Base L2
- Fixed supply with deflationary mechanism (burn on settlement)
- Initial distribution: team allocation, node operator incentives,
  community treasury, liquidity pool

Metering (building on M2 envelope counter):
- Each envelope already carries an envelope_count (uint64)
- Relay/entry node tracks envelopes processed per node per session
- Metering data signed by both node and consumer (non-repudiation)
- Off-chain metering, on-chain settlement

Payment channels:
- Consumer locks $GROOVE in a payment channel smart contract at session start
- Nodes claim proportional to work done:
  claim = envelopes_processed * layers_served * rate_per_envelope_per_layer
- Channels can be extended (add funds) or closed (settle)
- Dispute window: 24h after channel close for fraud proofs

Staking:
- Nodes stake $GROOVE to join the network
- Minimum stake scales with advertised capacity (prevents overcommitment)
- Slashing conditions:
  - Returning garbage output (proven via redundant compute)
  - Going offline mid-session without graceful shutdown
  - Failing to serve assigned layers after ASSIGNMENT_ACK
- Stake locked for cooldown period after unstaking (e.g., 7 days)

Price discovery:
- Nodes advertise their rate (tokens per envelope per layer)
- Consumer SDK selects cheapest pipeline meeting latency requirements
- Market dynamics: high demand -> nodes raise rates -> more nodes join
  -> rates stabilize

**Verification (the hard problem):**
- Redundant compute: randomly assign same work to 2+ nodes, compare results
- Spot-checking: consumer occasionally sends a known-answer probe
- Optimistic verification: assume honest, slash on proven fraud
  (cheaper than verifying every envelope)
- Reputation scores: nodes build track records over time; new nodes
  get less work until proven reliable

**Settlement architecture:**
- On-chain (Base L2): session open/close, stake/slash, token transfers,
  payment channel state, dispute resolution
- Off-chain: envelope counting, latency tracking, reputation updates
  (roll up to chain periodically via signed attestations)

**Implementation steps:**

Step 1 — Smart contracts (Solidity):
- GrooveToken.sol (ERC-20)
- NodeRegistry.sol (stake, register, slash)
- PaymentChannel.sol (open, claim, dispute, close)
- Deploy to Base Sepolia testnet first

Step 2 — Node integration:
- Node signs envelope counts with secp256k1 key (identity.py already has this)
- Node submits claims to PaymentChannel contract
- Node stakes on registration, auto-slash on protocol violations

Step 3 — Consumer SDK:
- Consumer locks tokens before session
- Consumer co-signs metering data
- Consumer can dispute fraudulent claims

Step 4 — Price oracle + routing:
- Nodes advertise rates via gossip (M3)
- Consumer picks optimal pipeline (cost vs. latency)
- Dynamic pricing based on network load

**Dependencies:** M3 (decentralized routing), M2 (envelope counter, node identity)

---

## M5 — Savant: Open-Source LLMs Trained on Groove Usage

Train open-source "Savant" LLMs on anonymized Groove usage patterns.
Nodes become domain specialists — a history tutor, a code assistant,
a medical advisor — each fine-tuned on the queries flowing through
the network. The models get smarter as the network grows.

**Goals:**

Data collection (opt-in):
- Users consent to anonymized usage data collection via Groove app settings
- Data stripped of PII before leaving the consumer
- Telemetry: query domains, response quality signals (thumbs up/down,
  follow-up patterns, session length), latency, model performance

Federated learning:
- Nodes fine-tune local shards on the inference traffic they process
- Gradient updates aggregated without exposing raw data
- Differential privacy: noise added to gradients before sharing
- Secure aggregation: no single party sees individual contributions

Specialization routing:
- Network learns which nodes excel at which domains
- Query classifier at entry node routes to best-fit specialist
- Domains: code, medical, legal, education, creative, multilingual
- Specialization score per node per domain (updated via reputation)

Model marketplace:
- Node operators publish specialized Savant model variants
- Consumers can request specific specialists (e.g., "savant-medical-v3")
- Marketplace integrated into Groove dashboard
- Rating/review system for model quality

Knowledge distillation:
- Large model outputs teach smaller local models
- Efficient specialists that run on modest hardware (8GB RAM)
- Progressive: start with Qwen2.5-0.5B, grow to 7B, 13B, 70B specialists

**Savant model releases:**
- savant-base: general purpose, trained on all Groove traffic
- savant-code: code generation specialist
- savant-medical: medical knowledge specialist
- savant-edu: educational tutor specialist
- All open-source, published on HuggingFace under Groove org

**Privacy guarantees:**
- No raw user data stored on-chain or in model weights
- Differential privacy with provable epsilon bounds
- User can delete their contribution (right to be forgotten)
- Regular privacy audits published to community

**Dependencies:** M4 (incentive layer funds training compute), M3 (distributed data collection)

---

## M6 — Production Hardening

The infrastructure that makes everything above reliable at global scale.

**Goals:**

Security:
- TLS everywhere (wss:// for all node/relay/consumer connections)
- mTLS between nodes (mutual authentication via secp256k1 certs)
- Rate limiting and DDoS protection at entry nodes
- Security audit of smart contracts before mainnet deployment

Operations:
- Prometheus metrics: node count, session count, latency percentiles,
  shard load times, envelope throughput, token settlement rates
- Grafana dashboards for network health
- PagerDuty/alerting for network-wide issues
- Structured JSON logging (already in place from M1)

Deployment:
- Docker images for node (CPU, CUDA, MPS variants)
- Kubernetes manifests with auto-scaling
- Helm chart for one-command cluster deployment
- CI/CD pipeline: automated testing, packaging, release tagging

Performance:
- Geographic routing: prefer nearby nodes for lower latency
- Model caching CDN: avoid redundant HuggingFace downloads
- Connection pooling and multiplexing optimizations
- Shard preloading hints (predictive loading based on demand patterns)

Reliability:
- Graceful degradation: network serves partial results if some nodes fail
- Circuit breakers: isolate failing nodes automatically
- Session migration: move active sessions between nodes on failure
- Preemption policy for rebalancing (deferred from M2)

**Dependencies:** Ongoing throughout M3-M5, critical before mainnet token launch

---

## Groove Dashboard Integration

The Groove desktop app (~/Desktop/groove) is the user-facing product.
Network features integrate into the dashboard so users can seamlessly
lend compute or consume intelligence without touching a terminal.

**Integration points (after M3 is stable):**

Node operator view:
- Toggle "Lend Compute" on/off from dashboard
- See assigned layers, active sessions, earnings in $GROOVE
- Hardware utilization gauges (CPU, GPU, RAM)
- Earnings history and payout schedule

Consumer view:
- Chat interface backed by Groove network (instead of centralized API)
- Model selector (general, code, medical, etc. — ties into M5 specialists)
- Session cost estimate before starting
- $GROOVE balance and top-up flow

Wallet integration:
- Base L2 wallet embedded in app (or MetaMask/Coinbase Wallet connect)
- Stake management for node operators
- Payment channel visualization for consumers
- Transaction history

Network status:
- Global node map (anonymized locations)
- Total compute capacity, active sessions
- Model coverage status (which models are fully covered)
- Network health indicators

**Implementation:** Handled by the Groove core team (~/Desktop/groove repo).
Groove-deploy exposes a local API/SDK that the dashboard consumes.
Integration starts after M3, iterates through M4 and M5.

---

## Current Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1 — Relay data plane | **Complete** | Tested with Qwen2.5-0.5B, multi-node verified |
| M2 — Dynamic layers | **Complete** | Scheduler, identity, rebalancing, envelope metering |
| M3 — Decentralized relay | Not started | Next up — DHT, gossip, embedded relay |
| M4 — Token economics | Not started | $GROOVE on Base L2, requires M3 |
| M5 — Savant training | Not started | Open-source specialist LLMs, requires M4 |
| M6 — Production hardening | Ongoing | TLS + auth should start with M3 |
| Dashboard integration | Not started | After M3 stable, via ~/Desktop/groove |

---

## Codebase Structure (post-M2)

```
groove-deploy/
  src/
    common/
      protocol.py          — message types, msgpack, envelope counter
      tensor_transfer.py   — binary tensor serialization
    relay/
      relay.py             — relay router, dynamic assignment, metering
      scheduler.py         — portable layer scheduler (no async deps)
    node/
      server.py            — compute node, on-demand shard loading
      shard_loader.py      — HuggingFace model shard loading
      kv_cache.py          — per-session KV cache management
      identity.py          — secp256k1 keypair, Ethereum addresses
    consumer/
      client.py            — inference client, pipeline walker
      speculative.py       — speculative decoding
      draft_model.py       — local draft model for speculation
  tests/
    test_scheduler.py      — scheduler algorithm tests (16 cases)
    test_identity.py       — keypair/signing tests (12 cases)
    test_dynamic_assignment.py — node handler tests (8 cases)
    test_dynamic_integration.py — end-to-end flow tests (2 cases)
    test_pipeline.py       — relay pipeline tests (9 cases)
    test_envelope_routing.py — envelope round-trip test
    test_shard_loading.py  — shard split verification
    test_speculative.py    — speculative decoding tests
    manual/
      mock_node.py         — echo node for smoke testing
      smoke_consumer.py    — standalone consumer smoke test
  requirements.txt
  setup.sh
  ROADMAP.md
  QUICKSTART.md
```

---

## Principles

1. **Open source first.** The protocol, node, and consumer are all open
   source. The network's value comes from participation, not lock-in.

2. **Contributor-friendly.** A node operator should go from zero to
   earning tokens in under 10 minutes. If setup is painful, people
   won't contribute.

3. **Incrementally decentralized.** Each milestone works standalone.
   M1 is useful without M2. M2 is useful without M3. You don't need
   the token to run a private network.

4. **Verify, don't trust.** Any system where strangers run your compute
   needs verification. This is non-negotiable before real tokens flow.

5. **Accessible to everyone.** The endgame is intelligence as a public
   utility. Design every decision to lower the barrier, not raise it.

6. **Base L2 native.** All on-chain logic lives on Base. Low fees, fast
   finality, EVM compatibility, strong ecosystem.

7. **Privacy by design.** User data never leaves their device unencrypted.
   Savant training uses differential privacy. Right to be forgotten is
   non-negotiable.

---

## Quick Reference for New Sessions

When starting a new session on this project:

1. Read this ROADMAP.md to understand the full vision and current status
2. Check the Current Status table above for what's done vs. what's next
3. The codebase is at ~/Desktop/groove-deploy/ (Python, asyncio, websockets)
4. The Groove dashboard is at ~/Desktop/groove/ (separate repo, separate team)
5. Node IDs are Ethereum addresses (secp256k1) — they map to Base L2 wallets
6. The scheduler in src/relay/scheduler.py is designed to be portable — it
   will move into the consensus layer in M3
7. Envelope counter is already in the wire format — M4 billing reads it
8. All tests: `pytest tests/ -v` from the groove-deploy root
