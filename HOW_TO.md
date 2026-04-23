# Groove Signal Service — Repo Guide

## Directory Layout

```
~/Desktop/groove-signal/   <- this repo (signal/relay service)
~/Desktop/groove-network/  <- sibling repo (node + consumer inference code)
```

These are **separate git repos** with separate remotes, tags, and deploy cycles.
Do NOT add the other repo as a second remote. That causes cross-push accidents.

## What This Repo Contains

- `src/signal/` — signal server (matcher, registry, scoring, dashboard)
- `src/relay/` — relay/envelope router
- `src/common/protocol.py` — signal-side protocol (superset of network protocol, includes matcher types)
- `src/common/tensor_transfer.py` — shared serialization (kept in sync manually)
- `Dockerfile` — production container
- `setup.sh` — deploy script

## Pushing Changes

```bash
cd ~/Desktop/groove-signal
git add <files>
git commit -m "description"
git push origin main
```

There is only one remote (`origin` = groove-signal). `git push` always does the right thing.

## Version Tags

The signal service auto-updates via cron by checking git tags. To release:

```bash
git tag -a v0.X.Y -m "v0.X.Y: description"
git push origin main --follow-tags
```

Tags MUST point to signal code commits. Never push tags from the network repo here.

## Shared Files (protocol.py, tensor_transfer.py)

`src/common/protocol.py` exists in both repos but has **diverged intentionally**:
- Signal version: includes matcher/signal-specific message types
- Network version: includes mesh/P2P/inference-specific message types
- Base message types (ENVELOPE, HEARTBEAT, REGISTER, etc.) are shared

If you change a base message type, update both repos. Signal-only or network-only
types stay in their respective repo.

`src/common/tensor_transfer.py` should stay identical. After changing it in one repo,
copy it to the other.

## Common Mistakes to Avoid

- Do NOT clone groove-network as a second remote in this directory
- Do NOT push network inference tags here (the cron will check out wrong code and crash)
- Do NOT run node/consumer code from this directory — use groove-network for that
