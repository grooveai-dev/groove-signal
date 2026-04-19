# Groove Roadmap

## Vision

A decentralized, crypto-incentivized global LLM inference network — PiperNet for AI.
Built on cypherpunk principles: permissionless, trustless, verifiable, open.

Anyone with a gaming rig, Mac Mini, or spare GPU contributes compute and earns
$GROOVE tokens. Anyone who needs intelligence buys it through the network. No
massive data centers, no single point of failure, no gatekeepers. Like Bitcoin
mining rewarded Proof of Work, Groove rewards Proof of Compute (PoC).

The Groove desktop app is the entry point. Open it, and you're either lending
compute power to the network or consuming it — seamlessly routed through smart
contracts on Base L2.

The network's secret weapon isn't raw scale — it's data. Every Groove user
who opts in contributes anonymized usage data: agent workflows, coding patterns,
integration chains, automation recipes, prompt-response pairs from frontier
models. This data is processed into high-quality training datasets and used to
pre-train open-source Savant LLMs from scratch. A 13B Savant model trained on
10TB of real Groove usage data can outperform a generic 400B model on the exact
tasks users care about. Smaller models that pack a punch in intelligence, run
fast on modest hardware, and get smarter as the network grows.

The flywheel: more Groove users -> more usage data -> better Savant models ->
more users attracted -> more data. The network gets smarter by existing.

**Three-tier participant model:**
```
Consumer — uses intelligence, pays $GROOVE
Node     — lends compute, earns $GROOVE for inference
Signal   — runs signaling relay + compute, earns routing fees + inference fees
```

**Tech stack:** Python (asyncio + websockets + PyTorch), Base L2 (EVM/Solidity),
Groove desktop app (Electron — separate repo at ~/Desktop/groove).
**Repo:** github.com/grooveai-dev/groove-network (public)

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
- HTTP GET /status endpoint on relay for network health (nodes, models, coverage)
- Consumer --json flag for structured JSON stdout (daemon subprocess parsing)
- setup.sh --json flag for structured install progress (daemon install flow)
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
- Test coverage gaps on relay status endpoint and consumer JSON output — M6

---

## M3 — Decentralized Signaling + Embedded Relay

Replace the central relay with a decentralized signaling network. Every node
embeds relay logic and binds to localhost only (never 0.0.0.0). Signal
operators run public-facing signaling services that broker connections
between consumers and nodes without ever touching inference traffic.

**Target model sizes: 7B-70B.** The network is optimized for Savant-class
models (7B-70B) pre-trained on Groove usage data, not generic 400B+ models.
A 13B Savant running on 1-2 nodes with sub-100ms latency and 30+ tokens/sec
beats a 400B model distributed across 32 nodes at 1 token/sec. Data quality
over parameter count.

**Three-tier architecture:**
```
                    ┌─────────────────────┐
                    │   Signal Operator   │
                    │  (matchmaker only)  │
                    │  scores & brokers   │
                    │  connections        │
                    └──────┬──────────────┘
                           │ outbound registration
              ┌────────────┼────────────┐
              v            v            v
         [Node A]     [Node B]     [Node C]
         127.0.0.1    127.0.0.1    127.0.0.1
         compute +    compute +    compute
         embedded     embedded
         relay        relay

    Consumer queries Signal -> gets ranked node list
    Consumer connects to best Node -> Node routes pipeline
    Signal never sees inference traffic
```

Consumer: uses the network, pays $GROOVE.
Node: lends compute power, earns inference fees. Binds to localhost only,
  connects outbound to signal(s). Embeds relay logic for pipeline routing.
Signal (Node+Signal): everything a Node does PLUS runs a public-facing
  signaling service. Earns routing fees on top of inference fees. Requires
  networking knowledge (port forwarding or VPS). The backbone of the network.

**Security model — nodes never expose ports:**
- Nodes bind to 127.0.0.1 only — no open ports, no 0.0.0.0
- Nodes connect OUTBOUND to signal services (same pattern as M1)
- Signals are the only public-facing infrastructure
- All node-to-node traffic encrypted end-to-end using secp256k1 identity keys
- Signals broker introductions but cannot read inference traffic
- A non-tech user clicking "Lend Compute" in the GUI is safe by default

**Gaussian decay scoring algorithm:**

When a consumer requests a connection, signals score every registered node:

```
score(node, consumer) =
    proximity_score * w1 +
    uptime_score    * w2 +
    compute_score   * w3 +
    load_score      * w4

proximity_score = exp(-distance_km² / (2 * σ_geo²))
  σ_geo tuned so nodes within 500km score high, 2000km+ decay sharply
  distance from IP geolocation (MaxMind GeoLite2)

uptime_score = exp(-downtime_events² / (2 * σ_up²))
  track disconnections over last 7 days
  perfectly stable node = 1.0

compute_score = normalize(vram_mb * gpu_weight + ram_mb * cpu_weight)
  RTX 4090 scores higher, but Mac Mini is still a valid node

load_score = 1.0 - (active_sessions / max_sessions)
  prefer nodes with capacity headroom
```

Consumer receives top-N ranked nodes, connects to the best available.
If that fails, falls back to second-best. Signal never learns what the
consumer asked — only that they needed a node.

**NAT traversal (connecting nodes behind firewalls):**
- Primary: STUN hole-punching (works for ~80% of NATs)
- Fallback: TURN-style relay through the signal operator (encrypted, signal
  cannot read the traffic — just forwarding bytes)
- Consumer and node negotiate connection via signal as the signaling channel
- Same pattern as WebRTC (proven at billions of connections per day)

**Implementation steps:**

Step 1 — Embedded relay + localhost binding:
- Merge relay logic into node process (node serves as both compute + router)
- Node binds to 127.0.0.1 only (secure by default)
- Config flag: --mode node | signal | node+signal
- Node mode: compute only, connects outbound to signals
- Signal mode: public-facing signaling service + routing
- Node+signal mode: both (the power user tier)

Step 2 — Signaling service:
- Lightweight service (~200 lines) that signal operators run
- Nodes connect outbound, register capabilities + approximate location
- Consumers query for best nodes using gaussian decay scoring
- Signal brokers WebSocket introductions (STUN/TURN)
- Signal never touches inference traffic — matchmaker only
- Initial deployment: Groove runs one signal at signal.groovedev.ai

Step 3 — Multi-signal consensus:
- Consumers query 3-5 signals simultaneously
- Take intersection/consensus of their rankings
- If 3/5 signals agree on a node ranking, it's probably honest
- A rogue signal's manipulated rankings get outvoted
- Consumers can flag inconsistencies for on-chain dispute

Step 4 — Signal gossip + node registry:
- Signals gossip node registrations with each other
- Each signal maintains a Merkle root of its full node registry
- Merkle proofs: any participant can verify a node exists (or was hidden)
- If signal A has 500 nodes and signal C only shows 50, that's
  provable dishonesty — evidence for slashing
- A consumer can query ANY signal and get a near-complete network view

Step 5 — Decentralized scheduling via signals:
- Move scheduler.py into the signal layer
- Signal operators run the scheduling algorithm for nodes registered with them
- For multi-signal pipelines: signals coordinate via gossip
- Leader election per model (highest-staked signal for that model)
- Fallback: nodes self-assign based on local view if no signal available

Step 6 — Geographic clustering + latency optimization:
- Nodes report approximate location (region/city, not precise GPS)
- Signals cluster nearby nodes automatically
- Pipeline assembly prefers geographically close nodes
- Latency probing: measure actual RTT between node pairs
- For a 70B model (8 nodes in same city): ~2ms per hop * 8 = 16ms per token
  = 60+ tokens/sec

Step 7 — Permissionless signal entry:
- Anyone can run a signal — no application, no approval
- Stake $GROOVE to register as signal operator (ties into M4)
- Signal operators earn routing fees proportional to connections brokered
- Bad signals lose nodes, lose consumers, lose fees — market-driven quality

**Signal operator security — consensus against bad actors:**

Inspired by Bitcoin: make cheating MORE EXPENSIVE than being honest.
You don't prevent bad behavior — you make it economically irrational.

Attack: Score manipulation (signal inflates scores for colluding nodes)
Defense: Multi-signal consensus. Consumer queries 3-5 signals, takes the
  majority ranking. A single rogue signal's manipulated scores get outvoted.
  Persistent inconsistency = flagged for slashing.

Attack: Eclipse attack (signal hides honest nodes from consumers)
Defense: Merkle root of node registry. Signals publish signed Merkle roots.
  If signal claims a node doesn't exist but another signal has a Merkle proof
  it does, that's provable fraud. Like Bitcoin's block headers — a chain of
  commitments you can verify.

Attack: Front-running (signal routes high-value requests to own nodes)
Defense: Commit-reveal routing. Signal commits to routing decision (hash)
  before seeing full request. Mismatch = provable fraud = slashing.
  Probabilistic auditing on 5% of requests keeps overhead low.

Attack: Sybil signals (one actor runs 100 signals to control routing)
Defense: Each signal requires $GROOVE stake. 100 signals = 100x capital at
  risk. Slashing on any one jeopardizes the whole operation. Like Bitcoin's
  51% attack cost — economically prohibitive. Stake age weighting: new signals
  with fresh stake get less traffic until proven reliable.

Attack: Surveillance (signal logs consumer-to-node connections)
Defense: Consumer queries multiple signals with slightly different parameters.
  No single signal sees the full picture. All consumer-to-node traffic
  encrypted end-to-end — signal cannot read inference content.

**Signal operator scoring (signals get scored too):**
```
signal_score =
    uptime              * w1 +
    connections_brokered * w2 +
    avg_latency_added   * w3 +
    stake_amount        * w4 +
    consistency_rating  * w5

consistency_rating = agreement with other signals' rankings
  (from multi-signal consensus audits)
```
Nodes prefer high-quality signals. Bad signals lose registrations, lose
consumers, lose fees, eventually get ignored. No central authority needed.

**Performance targets (M3):**
- 7B-13B Savant: 1-2 nodes, sub-100ms latency, 30+ tokens/sec
- 70B Savant: 4-8 nodes (same region), 5-15 tokens/sec
- Signal query to first token: <500ms
- Multi-signal consensus: <200ms (parallel queries)

**Key challenges:**
- NAT traversal reliability (STUN works ~80%, need TURN fallback)
- Gossip propagation latency between signals
- Sybil resistance before M4 staking is live (rate limiting + invite codes)
- Signal operator UX for non-trivial setup (port forwarding, VPS)

**Dependencies:** M2 (scheduler portability, node identity)

---

## M4 — Token Economics + Incentive Layer ($GROOVE on Base L2)

Crypto-economic layer that rewards all three tiers and creates a
self-sustaining marketplace for intelligence on Base L2.

**Cypherpunk economics:** Permissionless entry. No gatekeepers. Stake tokens,
contribute, earn. The protocol enforces honesty through economic incentives,
not trust. Like Bitcoin circa 2009 — anyone with hardware can participate
and earn, with mathematically guaranteed fairness.

**Why Base L2:**
- EVM-compatible (Solidity smart contracts)
- Low gas fees (fractions of a cent per tx)
- Fast finality (~2 seconds)
- Strong ecosystem (Coinbase, large user base)
- Team has deep Ethereum/blockchain development experience

**Low barrier to entry:** A Mac Mini running a 13B Savant model is a
first-class network participant earning $GROOVE. No 4090 required.

**Three-tier economics:**

```
┌──────────────────────────────────────────────────────────┐
│ Consumer                                                 │
│ - Pays $GROOVE per session                               │
│ - Locks tokens in payment channel at session start       │
│ - Co-signs metering data with node                       │
│ - Can dispute fraudulent claims (24h window)             │
└──────────────────────────────────────────────────────────┘
                          │
                    $GROOVE flows down
                          │
┌──────────────────────────────────────────────────────────┐
│ Signal Operator (Node+Signal tier)                       │
│ - Stakes: 5000+ $GROOVE (higher trust = more skin)       │
│ - Earns: 2-5% routing fee on every session they broker   │
│ - Earns: inference fees for compute work (same as Node)  │
│ - Slashing: score manipulation, eclipse attacks,         │
│   front-running, going offline, inconsistent routing     │
│ - Cooldown: 14 days after unstaking                      │
└──────────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────┐
│ Node Operator                                            │
│ - Stakes: 1000+ $GROOVE                                  │
│ - Earns: per-envelope * layers_served * rate             │
│ - Slashing: garbage output, mid-session dropout,         │
│   failing to serve after ASSIGNMENT_ACK                  │
│ - Cooldown: 7 days after unstaking                       │
└──────────────────────────────────────────────────────────┘
```

**Token contract:**
- ERC-20 $GROOVE token on Base L2
- Fixed supply with deflationary mechanism (burn on settlement)
- Initial distribution: team allocation, node operator incentives,
  community treasury, liquidity pool

**Metering (building on M2 envelope counter):**
- Each envelope already carries an envelope_count (uint64)
- Entry node / signal tracks envelopes processed per node per session
- Metering data signed by both node and consumer (non-repudiation)
- Off-chain metering, on-chain settlement

**Payment channels:**
- Consumer locks $GROOVE in a payment channel smart contract at session start
- Nodes claim proportional to work done:
  claim = envelopes_processed * layers_served * rate_per_envelope_per_layer
- Signal operator claims routing fee: small percentage of session total
- Channels can be extended (add funds) or closed (settle)
- Dispute window: 24h after channel close for fraud proofs

**Staking:**
- Node stake: minimum 1000 $GROOVE, scales with advertised capacity
- Signal stake: minimum 5000 $GROOVE (higher trust requirement)
- Slashing conditions (Node):
  - Returning garbage output (proven via redundant compute)
  - Going offline mid-session without graceful shutdown
  - Failing to serve assigned layers after ASSIGNMENT_ACK
- Slashing conditions (Signal — additional):
  - Score manipulation (proven via multi-signal consensus audit)
  - Eclipse attack (proven via Merkle root inconsistency)
  - Front-running (proven via commit-reveal mismatch)
  - Routing to offline/nonexistent nodes
- Stake locked for cooldown: 7 days (Node), 14 days (Signal)

**Price discovery:**
- Nodes advertise their rate (tokens per envelope per layer)
- Signals advertise their routing fee percentage
- Consumer SDK selects cheapest pipeline meeting latency requirements
- Market dynamics: high demand -> rates rise -> more operators join
  -> rates stabilize -> self-regulating equilibrium

**Verification (the hard problem — Bitcoin-inspired):**
- Redundant compute: randomly assign same work to 2+ nodes, compare results
- Spot-checking: consumer occasionally sends a known-answer probe
- Optimistic verification: assume honest, slash on proven fraud
  (cheaper than verifying every envelope — like Bitcoin's SPV)
- Reputation scores: nodes and signals build track records over time;
  new participants get less work until proven reliable
- Multi-signal consensus: consumers cross-check routing decisions,
  flag inconsistencies for on-chain dispute
- All verifiable: anyone can audit, no trusted third party needed

**Settlement architecture:**
- On-chain (Base L2): session open/close, stake/slash, token transfers,
  payment channel state, dispute resolution, signal registry,
  Merkle root commitments, reputation snapshots
- Off-chain: envelope counting, latency tracking, routing decisions,
  reputation updates, signal gossip
  (roll up to chain periodically via signed attestations)

**Decentralization path for signals:**

Phase 1 (beta): Groove runs the signaling service on signal.groovedev.ai.
  Centralized but functional. No staking required yet.

Phase 2 (M3 complete): Open signaling as an open-source package. Tech-savvy
  operators deploy their own. Nodes register with multiple signals for
  redundancy. No staking yet — invite-only signal operators.

Phase 3 (M4 launch): Signal operator staking on Base L2. $GROOVE fees flow
  to signals proportional to connections brokered. Quality scores on-chain.
  Permissionless entry — anyone can stake and become a signal.

Phase 4 (mature): Signals gossip with each other, sharing node registrations.
  Consumer can query ANY signal and get a global view. Fully decentralized.
  No single signal has to be online for the network to function.
  Bootstrap list is a smart contract — anyone registers by staking.

**Implementation steps:**

Step 1 — Smart contracts (Solidity):
- GrooveToken.sol (ERC-20)
- NodeRegistry.sol (stake, register, slash)
- SignalRegistry.sol (signal stake, register, routing fee claim, slash)
- PaymentChannel.sol (open, claim, dispute, close)
- Deploy to Base Sepolia testnet first

Step 2 — Node integration:
- Node signs envelope counts with secp256k1 key (identity.py already has this)
- Node submits claims to PaymentChannel contract
- Node stakes on registration, auto-slash on protocol violations

Step 3 — Signal integration:
- Signal operator stakes via SignalRegistry contract
- Signal submits Merkle root commitments on-chain (periodic)
- Signal claims routing fees from PaymentChannel
- Multi-signal consensus audit results posted on-chain

Step 4 — Consumer SDK:
- Consumer locks tokens before session
- Consumer co-signs metering data
- Consumer can dispute fraudulent claims
- Consumer queries multiple signals, flags inconsistencies

Step 5 — Price oracle + routing:
- Nodes and signals advertise rates via gossip
- Consumer picks optimal pipeline (cost vs. latency vs. reputation)
- Dynamic pricing based on network load
- Geographic arbitrage: prices vary by region based on supply/demand

**Dependencies:** M3 (decentralized signaling), M2 (envelope counter, node identity)

---

## M5 — Savant: Open-Source LLMs Pre-Trained on Groove Usage Data

The core insight: you don't need a 400B model if you have 10TB of
high-quality, task-specific data. Savant models are 7B-70B parameter LLMs
pre-trained from scratch on real Groove usage data — agent workflows,
coding sessions, integration patterns, automation recipes, and
prompt-response pairs from frontier models. The result is a compact model
that punches far above its weight class on the tasks Groove users actually
perform.

**The data flywheel:**
```
More Groove users
  -> more opt-in usage data
    -> better Savant training datasets
      -> smarter Savant models
        -> better Groove experience
          -> more users attracted
            -> more data (repeat)
```

**Why pre-training, not fine-tuning:**
Fine-tuning a base model (LoRA, QLoRA) adds a thin layer of task knowledge
on top of generic weights. Pre-training bakes task knowledge INTO the weights
from the ground up. When you have 10TB of curated Groove data, pre-training
produces a model that natively understands agent orchestration, code
generation patterns, tool use, multi-step reasoning chains, and integration
workflows — not a generic model with a task-specific veneer.

This is how a 13B Savant outperforms a 400B generic model on Groove-relevant
tasks. The data does the heavy lifting, not the parameter count.

**Data pipeline:**

Collection (opt-in, privacy-first):
- Users toggle "Contribute to Savant" in Groove app settings
- Data captured at the consumer/app level, never from node operators
- Raw collection includes: prompts, responses, tool calls, agent workflows,
  code diffs, integration chains, user feedback signals (thumbs up/down,
  follow-up patterns, session length, task completion)
- All data stripped of PII before leaving the user's device:
  - Named entity recognition to remove names, emails, API keys
  - Path/URL sanitization (replace with placeholders)
  - Code anonymization (rename project-specific identifiers)

Processing and curation:
- Centralized data pipeline (not on-network — training data is curated
  by the Groove team, not distributed across nodes)
- Quality filtering: discard low-signal sessions (very short, abandoned,
  thumbs-down responses)
- Deduplication: remove near-duplicate prompts and responses
- Domain tagging: classify each sample (code, automation, integration,
  analysis, creative, general)
- Curriculum design: balance domain distribution for training
- Format into standard pre-training corpus (tokenized, shuffled, sharded)

**Savant model tiers:**

savant-7b:
- Target: single-node inference on any machine with 8GB+ RAM
- Focus: fast, general-purpose, covers 80% of Groove tasks
- Training: full pre-training run on curated Groove data
- Performance target: match GPT-4-mini on Groove-specific benchmarks

savant-13b:
- Target: single-node on 16GB+ RAM or 1-2 nodes on network
- Focus: code generation, complex agent workflows
- Training: pre-training + extended training on code-heavy subset
- Performance target: match GPT-4 on coding and multi-step agent tasks

savant-70b:
- Target: 4-8 nodes on network, power users and complex reasoning
- Focus: frontier-level intelligence for hardest tasks
- Training: pre-training on full dataset, longest training run
- Performance target: competitive with Claude/GPT-4 on Groove tasks

All models open-source, published on HuggingFace under Groove org.

**Continuous improvement:**
- New Savant versions trained quarterly as data grows
- Each version benchmarked against previous + frontier models
- A/B testing: route a fraction of network traffic to new Savant
  versions, measure user satisfaction signals
- Community feedback loop: users rate Savant responses, best/worst
  examples feed back into training data curation

**Network integration:**
- Savant models are the DEFAULT models served by the Groove network
- Node operators download Savant shards just like any HuggingFace model
- Query routing: signal classifies query domain, routes to nodes
  running the best-fit Savant variant
- Model marketplace: community can publish specialized fine-tunes of
  Savant base models (savant-legal, savant-medical, etc.)

**Privacy guarantees:**
- No raw user data stored on-chain or shared with node operators
- PII stripping happens on-device before data leaves the user
- Differential privacy applied during training (provable epsilon bounds)
- User can delete their contribution at any time (right to be forgotten)
- Regular privacy audits published to community
- Training data pipeline is auditable (open-source tooling)

**Dependencies:** M4 (token economics funds training compute via network),
M3 (network serves Savant models), Groove app (data collection UI)

---

## M6 — Production Hardening

The infrastructure that makes everything above reliable at global scale.

**Goals:**

Security:
- TLS everywhere (wss:// for all connections)
- mTLS between nodes (mutual authentication via secp256k1 certs)
- Challenge-response node authentication (signal verifies node owns
  claimed keypair — identity.py sign/verify already supports this)
- Rate limiting and DDoS protection at signal operators
- Security audit of smart contracts before mainnet deployment
- Input validation on all protocol messages (msgpack size limits,
  capability value bounds, tensor dimension checks)
- Signal operator security hardening (TLS termination, abuse detection)

Operations:
- Prometheus metrics: node count, signal count, session count, latency
  percentiles, shard load times, envelope throughput, routing fees,
  token settlement rates
- Grafana dashboards for network health
- PagerDuty/alerting for network-wide issues
- Structured JSON logging (already in place from M1)

Deployment:
- Docker images for node (CPU, CUDA, MPS variants)
- Docker image for signal operator (lightweight, no GPU needed)
- Kubernetes manifests with auto-scaling
- Helm chart for one-command cluster deployment
- CI/CD pipeline: automated testing, packaging, release tagging

Performance:
- Continuous batching (vLLM-style): nodes serve 10-50 concurrent
  sessions by batching forward passes, not processing sequentially
- INT4/INT8 quantization: cut layer memory 2-4x, let fewer nodes hold
  more layers, reduce pipeline hops, improve latency
- Model caching CDN: avoid redundant HuggingFace/Savant downloads
- Connection pooling and multiplexing optimizations
- Shard preloading hints (predictive loading based on demand patterns)

Reliability:
- Graceful degradation: network serves partial results if some nodes fail
- Circuit breakers: isolate failing nodes automatically
- Session migration: move active sessions between nodes on failure
- Preemption policy for rebalancing (deferred from M2)
- Rebalance atomicity: rollback if some nodes reject new assignments
- Signal redundancy: nodes registered with 3+ signals, auto-failover

**Dependencies:** Ongoing throughout M3-M5, critical before mainnet token launch

---

## Groove Dashboard Integration

The Groove desktop app (~/Desktop/groove) is the user-facing product.
Network features integrate into the dashboard so users can seamlessly
lend compute or consume intelligence without touching a terminal.

**Beta access flow:**
- User enters invite code in Groove app
- Unlocks Network view with "Install Network Package" button
- Groove daemon clones grooveai-dev/groove-network to ~/.groove/network/
- Runs setup.sh --json, streams install progress to GUI
- Once installed: full node operator + network status + consumer UI

**Integration points (iterates through M3-M5):**

Node operator view:
- Toggle "Lend Compute" on/off from dashboard
- Node binds to localhost only — safe by default, no networking knowledge needed
- See assigned layers, active sessions, earnings in $GROOVE
- Hardware utilization gauges (CPU, GPU, RAM)
- Earnings history and payout schedule

Signal operator view (advanced, opt-in):
- "Run Signal" toggle for power users who want to earn routing fees
- Requires port forwarding or VPS — dashboard shows setup guide
- Signal status: registered nodes, connections brokered, routing fees earned
- Merkle root publication status
- Signal score and ranking among other signals

Consumer view:
- Chat interface backed by Groove network (alongside centralized providers)
- Model selector (savant-7b, savant-13b, savant-70b, plus specialists)
- Session cost estimate before starting
- $GROOVE balance and top-up flow
- Multi-signal health indicator (how many signals responded)

Savant data contribution:
- "Contribute to Savant" toggle in settings (opt-in)
- Dashboard shows: data contributed, anonymization status, which Savant
  version your data trained, impact metrics
- Delete contribution button (right to be forgotten)

Wallet integration:
- Base L2 wallet embedded in app (or MetaMask/Coinbase Wallet connect)
- Stake management for node and signal operators
- Payment channel visualization for consumers
- Transaction history
- Slashing alerts for operators

Network status:
- Global node map (anonymized locations)
- Total compute capacity, active sessions, signal count
- Model coverage status (which Savant models are fully covered)
- Network health indicators
- Signal operator leaderboard (by score)

**Implementation:** Handled by the Groove core team (~/Desktop/groove repo).
Groove-deploy exposes a local API/SDK that the dashboard consumes.
Cross-team communication via ~/Desktop/groove-comms/.

---

## Current Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1 — Relay data plane | **Complete** | Tested with Qwen2.5-0.5B, multi-node verified |
| M2 — Dynamic layers | **Complete** | Scheduler, identity, rebalancing, envelope metering |
| Dashboard integration | **In progress** | Beta install flow, node operator UI, network status |
| M3 — Decentralized signaling | **In progress** | Signal service built + deployed to signal.groovedev.ai |
| M4 — Token economics | Not started | $GROOVE on Base L2, three-tier staking, requires M3 |
| M5 — Savant pre-training | Not started | Data pipeline + 7B/13B/70B models, requires M4 |
| M6 — Production hardening | Ongoing | Security fixes in progress, batching + quant for M3+ |

**M3 progress (as of 2026-04-19):**
- [x] Step 2 — Signal service built and deployed (grooveai-dev/groove-signal v0.1.0)
  - Server (1164 lines): WebSocket + HTTP, rate limiting, TLS support
  - Gaussian decay scoring (326 lines): haversine, proximity, uptime, compute, load
  - Node registry (210 lines): Merkle root commitments (SHA256 binary tree)
  - Consumer matcher (140 lines): pipeline assembly with scoring
  - Scheduler (201 lines): portable from groove-network
  - Dockerfile, 58 tests passing
  - Deployed to signal.groovedev.ai (WebSocket 8770, HTTP 8771 via nginx)
- [x] --signal flag wired up in node server.py and consumer client.py
  - Mutually exclusive with --relay (legacy), auto-enables TLS
  - Consumer emits signal_connected and matched JSON events
- [ ] Step 1 — Embedded relay + localhost binding (node embeds relay logic)
- [ ] Step 3 — Multi-signal consensus
- [ ] Step 4 — Signal gossip + node registry sync
- [ ] Step 5 — Decentralized scheduling via signals
- [ ] Step 6 — Geographic clustering + latency optimization
- [ ] Step 7 — Permissionless signal entry

---

## Network Performance Summary

**Why Savant models change the game:**

The decentralized network has an inherent latency cost: each hop between
nodes adds 10-50ms depending on geography. For a 400B model across 32 nodes,
that's ~1 second per token — unusable for interactive chat.

But Groove doesn't need 400B generic models. Savant models pre-trained on
real Groove usage data achieve frontier-level performance at 7B-70B scale.
This means:

| Model | Nodes per pipeline | Latency per token | Tokens/sec | Use case |
|-------|-------------------|-------------------|------------|----------|
| savant-7b | 1 (single machine) | <30ms | 30-60 | Most Groove tasks |
| savant-13b | 1-2 nodes | 30-80ms | 12-30 | Code, complex agents |
| savant-70b | 4-8 nodes (same region) | 50-200ms | 5-15 | Frontier reasoning |

Data quality > parameter count. 10TB of curated Groove data makes a 13B
model smarter than a 400B generic model on Groove-relevant tasks.

**Bottlenecks and solutions (addressed across M3-M6):**
- Pipeline latency: geographic clustering via gaussian scoring, fewer fatter hops
- Concurrent sessions: continuous batching (vLLM-style) — 10-50x throughput
- Model memory: INT4 quantization — 4x reduction, more layers per node
- Model downloads: caching CDN for Savant shards
- Smart routing: gaussian decay scoring, domain-based model selection
- NAT traversal: STUN hole-punching (80%), TURN fallback (20%)

**Scale math (target: 100k nodes, 500k daily users):**
- savant-13b on 100k nodes: each node runs full model, 100k concurrent sessions
- With continuous batching (20x): 2M concurrent sessions — more than enough
- Mac Mini with 16GB RAM: runs savant-7b or savant-13b as first-class node
- Low barrier to entry + real $GROOVE earnings = organic network growth

---

## Codebase Structure (post-M3 signal service)

Three repos under grooveai-dev:

```
groove-network/  (github.com/grooveai-dev/groove-network)
  src/
    common/
      protocol.py          — message types, msgpack, envelope counter, M3 signal types
      tensor_transfer.py   — binary tensor serialization
    relay/
      relay.py             — relay router, dynamic assignment, metering, HTTP /status
      scheduler.py         — portable layer scheduler (no async deps)
    node/
      server.py            — compute node, --signal flag, on-demand shard loading
      shard_loader.py      — HuggingFace model shard loading
      kv_cache.py          — per-session KV cache management
      identity.py          — secp256k1 keypair, Ethereum addresses
    consumer/
      client.py            — inference client, --signal flag, --json output
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
  setup.sh                 — install script (supports --json for daemon integration)
  ROADMAP.md
  QUICKSTART.md

groove-signal/  (github.com/grooveai-dev/groove-signal, deployed to signal.groovedev.ai)
  src/
    signal/
      server.py            — WebSocket + HTTP signal server (1164 lines)
      scoring.py           — gaussian decay scoring, haversine, IP geolocation
      registry.py          — node registry with Merkle root commitments
      matcher.py           — consumer matcher, pipeline assembly
    common/
      protocol.py          — shared protocol (copy from groove-network)
    relay/
      scheduler.py         — portable scheduler (copy from groove-network)
  tests/                   — 58 tests
  Dockerfile               — python:3.12-slim, ports 8770+8771
  requirements.txt
  setup.sh
  README.md

groove/  (github.com/grooveai-dev/groove — Electron desktop app, separate team)
  — Beta invite code flow, network view, node toggle, consumer chat
  — Communicates via ~/Desktop/groove-comms/
```

---

## Principles

1. **Cypherpunk ethos.** Permissionless, trustless, verifiable, open.
   No gatekeepers, no applications, no approval processes. Stake tokens
   and participate. The protocol enforces honesty through economics,
   not authority. Bringing crypto back to its 2008-2017 glory days —
   built for the tech, not the speculation.

2. **Data over parameters.** A 13B model trained on 10TB of real usage data
   beats a 400B generic model on the tasks that matter. Invest in data
   quality and curation, not raw scale.

3. **Open source first.** The protocol, node, signal, consumer, and Savant
   models are all open source. The network's value comes from participation,
   not lock-in.

4. **Contributor-friendly.** A node operator should go from zero to earning
   tokens in under 10 minutes. A Mac Mini is a first-class network
   participant. If setup is painful, people won't contribute.

5. **Secure by default.** Nodes bind to localhost. No open ports for
   non-tech users. Signals are the only public-facing infrastructure,
   run by operators who understand networking. End-to-end encryption
   on all inference traffic.

6. **Incrementally decentralized.** Each milestone works standalone.
   M1 is useful without M2. M2 is useful without M3. You don't need
   the token to run a private network.

7. **Verify, don't trust.** Multi-signal consensus, Merkle proofs,
   commit-reveal routing, redundant compute, slashing. Every claim
   is independently verifiable. No trusted third parties.

8. **Accessible to everyone.** The endgame is intelligence as a public
   utility. Design every decision to lower the barrier, not raise it.

9. **Base L2 native.** All on-chain logic lives on Base. Low fees, fast
   finality, EVM compatibility, strong ecosystem.

10. **Privacy by design.** User data never leaves their device unencrypted.
    PII stripped on-device. Savant training uses differential privacy.
    Right to be forgotten is non-negotiable.

---

## Quick Reference for New Sessions

When starting a new session on this project:

1. Read this ROADMAP.md to understand the full vision and current status
2. Check the Current Status table above for what's done vs. what's next
3. Three repos under grooveai-dev:
   - groove-network (~/Desktop/groove-deploy/) — node + consumer + relay (Python)
   - groove-signal (~/Desktop/groove-deploy/default/groove-signal/) — signal service (Python)
   - groove (~/Desktop/groove/) — Electron desktop app (separate team)
4. Cross-team communication: ~/Desktop/groove-comms/ (requests + responses)
5. Signal service LIVE at signal.groovedev.ai (WebSocket 8770, HTTP 8771 via nginx)
6. Three-tier model: Consumer (pays) / Node (compute) / Signal (routing + compute)
7. Nodes bind to 127.0.0.1 ONLY — never 0.0.0.0. Security by default.
8. Node/consumer use --signal flag for production (e.g. --signal signal.groovedev.ai)
9. --relay flag preserved for legacy/local dev
10. Signal operators are the only public-facing infrastructure
11. Node IDs are Ethereum addresses (secp256k1) — they map to Base L2 wallets
12. The scheduler in src/relay/scheduler.py is portable — already copied into signal
13. Envelope counter is already in the wire format — M4 billing reads it
14. Network is optimized for Savant models (7B-70B), not giant 400B+ models
15. The competitive advantage is DATA (Groove usage), not model size
16. Gaussian decay scoring for node selection: proximity, uptime, compute, load
17. Multi-signal consensus prevents routing manipulation (Bitcoin-inspired)
18. All tests (groove-network): `pytest tests/ -v` from groove-deploy root
19. All tests (groove-signal): `pytest tests/ -v` from groove-signal root
