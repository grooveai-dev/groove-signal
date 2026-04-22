# Coturn TURN Server Deployment

TURN relay fallback for nodes behind symmetric NATs where STUN-only P2P fails.

## Quick Start

```bash
# Replace <GROOVE_TURN_SECRET> in turnserver.conf with your shared secret,
# then run coturn in Docker:
docker run -d --name coturn --network=host \
  -v $(pwd)/turnserver.conf:/etc/coturn/turnserver.conf \
  coturn/coturn
```

## Production

Deploy to 2-3 VPS regions (e.g. US-West, US-East, EU) on $5/month instances.
Set the same `static-auth-secret` on all instances and in the relay's
`GROOVE_TURN_SECRET` environment variable (or `--turn-secret` flag).

Ports to open: 3478/tcp+udp, 5349/tcp+udp, 49152-65535/udp.
