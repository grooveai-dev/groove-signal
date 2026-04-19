"""Gaussian-decay scoring for matching consumers to Groove nodes.

Signals score every registered node against every incoming consumer query
along four dimensions — proximity, uptime, compute, load — then combine
the dimensions with a configurable weight vector.

The scoring surface is intentionally smooth (Gaussian decay rather than
hard thresholds) so small changes in node/consumer state produce small
changes in rank. This reduces oscillation during rebalancing and
minimizes cliff-edge behavior where a node with "almost enough" capacity
is ranked drastically worse than one with "just enough".

Pure-logic module — no asyncio, no websockets, no I/O. Portable into the
M3 consensus layer without modification.
"""

from __future__ import annotations

import math
from typing import Iterable


# Gaussian widths. Hand-tuned for real-world geographic and uptime distributions.
SIGMA_GEO_KM = 800.0       # nodes within ~500km score high; ~2000km+ decay sharply
SIGMA_UPTIME = 3.0          # 0-1 disconnects = ~1.0; 5+ = low

DEFAULT_WEIGHTS = {
    "proximity": 0.35,
    "uptime": 0.25,
    "compute": 0.20,
    "load": 0.20,
}

# Neutral defaults when a dimension cannot be computed (e.g. no geolocation).
NEUTRAL_PROXIMITY = 0.5
NEUTRAL_UPTIME = 0.7    # new nodes get benefit of the doubt, but not perfect

# Floor for compute score so weak-but-valid nodes still have a chance.
COMPUTE_FLOOR = 0.1

# Clamp for self-reported capability values to prevent score gaming.
_MAX_PLAUSIBLE_VRAM_MB = 160_000
_MAX_PLAUSIBLE_RAM_MB = 4_000_000


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _gaussian(x: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-(x ** 2) / (2.0 * sigma ** 2))


def proximity_score(
    node_location: dict | None,
    consumer_location: dict | None,
    sigma_km: float = SIGMA_GEO_KM,
) -> float:
    """Gaussian decay on geographic distance. Neutral when location missing."""
    if not node_location or not consumer_location:
        return NEUTRAL_PROXIMITY
    try:
        dist = haversine_km(
            float(node_location["lat"]), float(node_location["lon"]),
            float(consumer_location["lat"]), float(consumer_location["lon"]),
        )
    except (KeyError, TypeError, ValueError):
        return NEUTRAL_PROXIMITY
    return _gaussian(dist, sigma_km)


def uptime_score(
    downtime_events: int | None,
    sigma: float = SIGMA_UPTIME,
) -> float:
    """Gaussian decay on disconnection count over a rolling 7-day window.

    New nodes with no history (None) receive a neutral-but-tentative score.
    A node with zero disconnects scores 1.0; 5+ disconnects scores low.
    """
    if downtime_events is None:
        return NEUTRAL_UPTIME
    # Negative counts are nonsensical; clamp.
    events = max(0, int(downtime_events))
    return _gaussian(float(events), sigma)


def _effective_capacity_mb(caps: dict) -> float:
    """Memory effectively usable for shard layers, in MB.

    GPU nodes weighted by VRAM at 1.0; CPU nodes weighted by RAM at 0.5
    because RAM-only inference is meaningfully slower. Matches the
    scheduler's bias so scoring and scheduling agree on node ordering.
    """
    device = (caps.get("device") or "cpu").lower()
    vram = min(max(float(caps.get("vram_mb") or 0), 0), _MAX_PLAUSIBLE_VRAM_MB)
    ram = min(max(float(caps.get("ram_mb") or 0), 0), _MAX_PLAUSIBLE_RAM_MB)
    if device in ("cuda", "gpu", "rocm", "mps") and vram > 0:
        return vram
    return ram * 0.5


def compute_score(
    node_caps: dict,
    max_capacity_mb: float,
    floor: float = COMPUTE_FLOOR,
) -> float:
    """Normalize a node's effective capacity against the registry max.

    Floor ensures even low-capacity nodes remain in the ranking rather
    than dropping to zero.
    """
    cap = _effective_capacity_mb(node_caps or {})
    if max_capacity_mb <= 0:
        return floor
    raw = cap / max_capacity_mb
    if raw < 0:
        raw = 0.0
    if raw > 1:
        raw = 1.0
    return max(floor, raw)


def _max_sessions_for(caps: dict) -> int:
    """Estimate a node's session capacity from its capabilities.

    A GPU can sustain roughly one session per 500 MB of VRAM; a CPU one
    per 1000 MB of RAM. These are coarse heuristics and should be
    recalibrated once real telemetry is available.
    """
    device = (caps.get("device") or "cpu").lower()
    vram = float(caps.get("vram_mb") or 0)
    ram = float(caps.get("ram_mb") or 0)
    if device in ("cuda", "gpu", "rocm", "mps") and vram > 0:
        return max(1, int(vram / 500))
    return max(1, int(ram / 1000))


def load_score(caps: dict, active_sessions: int) -> float:
    """1.0 - (active / max); clamped to [0, 1].

    Nodes with no active sessions score 1.0; fully saturated nodes score 0.
    """
    max_sess = _max_sessions_for(caps or {})
    busy = max(0, int(active_sessions or 0))
    raw = 1.0 - (busy / max_sess)
    if raw < 0:
        return 0.0
    if raw > 1:
        return 1.0
    return raw


def _normalize_weights(weights: dict | None) -> dict:
    """Fill in defaults for any missing weight keys and renormalize to sum=1.

    Accepting arbitrary weight vectors without renormalization would mean
    composite scores vary wildly in magnitude between queries; renormalizing
    keeps scores in [0, 1] regardless of the operator's weight choices.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k in DEFAULT_WEIGHTS:
            if k in weights:
                w[k] = max(0.0, float(weights[k]))
    total = sum(w.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in w.items()}


def score_node(
    node: dict,
    consumer: dict,
    weights: dict | None = None,
    max_capacity_mb: float | None = None,
) -> float:
    """Composite score for a node given a consumer request.

    node:  {node_id, capabilities, location, downtime_events, active_sessions}
    consumer: {location, requirements?}
    weights: optional override of DEFAULT_WEIGHTS
    max_capacity_mb: optional pre-computed registry max (avoids re-scan
                     per-call when ranking many nodes)
    """
    w = _normalize_weights(weights)
    caps = node.get("capabilities") or {}

    if max_capacity_mb is None:
        max_capacity_mb = max(1.0, _effective_capacity_mb(caps))

    p = proximity_score(node.get("location"), consumer.get("location"))
    u = uptime_score(node.get("downtime_events"))
    c = compute_score(caps, max_capacity_mb)
    ll = load_score(caps, node.get("active_sessions", 0))

    return (
        p * w["proximity"]
        + u * w["uptime"]
        + c * w["compute"]
        + ll * w["load"]
    )


def _compute_max_capacity(nodes: Iterable[dict]) -> float:
    best = 0.0
    for n in nodes:
        cap = _effective_capacity_mb(n.get("capabilities") or {})
        if cap > best:
            best = cap
    return max(1.0, best)


def rank_nodes(
    nodes: list[dict],
    consumer: dict,
    top_n: int = 10,
    weights: dict | None = None,
) -> list[dict]:
    """Return top-N nodes sorted by score descending, with score attached.

    Each output dict is a shallow copy of the input with an added "score"
    field; the caller's node dicts are not mutated.
    """
    if not nodes:
        return []
    max_cap = _compute_max_capacity(nodes)
    scored = []
    for n in nodes:
        s = score_node(n, consumer, weights=weights, max_capacity_mb=max_cap)
        entry = dict(n)
        entry["score"] = s
        scored.append(entry)
    scored.sort(key=lambda e: e["score"], reverse=True)
    if top_n > 0:
        return scored[:top_n]
    return scored


# ---------------------------------------------------------------------------
# IP geolocation.
# ---------------------------------------------------------------------------
# geoip2 is an optional dependency. If unavailable, we return None and the
# caller falls back to neutral proximity. For beta this is acceptable; the
# GeoLite2 database download is a deployment concern, not a code concern.
try:  # pragma: no cover - import-path branch
    import geoip2.database  # type: ignore
    import geoip2.errors  # type: ignore
    _HAS_GEOIP2 = True
except ImportError:  # pragma: no cover
    _HAS_GEOIP2 = False


_geoip_reader = None
_geoip_db_path: str | None = None


def configure_geoip(db_path: str | None) -> None:
    """Point the module at a MaxMind GeoLite2-City .mmdb file.

    Called once at signal startup. If db_path is None or geoip2 is not
    installed, IP lookups silently return None.
    """
    global _geoip_reader, _geoip_db_path
    _geoip_db_path = db_path
    _geoip_reader = None  # lazy-open on first lookup


def _get_reader():  # pragma: no cover - I/O
    global _geoip_reader
    if _geoip_reader is not None:
        return _geoip_reader
    if not _HAS_GEOIP2 or not _geoip_db_path:
        return None
    try:
        _geoip_reader = geoip2.database.Reader(_geoip_db_path)
    except (OSError, ValueError):
        _geoip_reader = None
    return _geoip_reader


def estimate_location_from_ip(ip_address: str) -> dict | None:
    """Approximate {lat, lon, city, region, country} from an IP address.

    Returns None on any failure (no DB configured, private IP, geoip2 not
    installed, record missing). Callers should treat None as "no location
    information" and use neutral proximity scoring — we never error-out
    on a geolocation miss.
    """
    if not ip_address:
        return None
    reader = _get_reader()
    if reader is None:
        return None
    try:  # pragma: no cover - I/O
        response = reader.city(ip_address)
    except Exception:
        return None
    lat = getattr(response.location, "latitude", None)
    lon = getattr(response.location, "longitude", None)
    if lat is None or lon is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "city": getattr(response.city, "name", None) or "",
        "region": (
            getattr(response.subdivisions.most_specific, "name", None) or ""
        ),
        "country": getattr(response.country, "name", None) or "",
    }
