"""Catalog sources (Space-Track primary, CelesTrak fallback), N2YO client and orbit helpers.

Secrets come from backend/.env (SPACETRACK_USER, SPACETRACK_PASS, N2YO_API_KEY, TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID). They are never logged, never put in error messages and never returned to the frontend.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sgp4.api import Satrec

log = logging.getLogger("kesslerwatch")
# httpx logs full request URLs at INFO; the N2YO key travels in the URL, so keep these quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
SAMPLE_FILE = CACHE_DIR / "sample_tles.txt"
USER_AGENT = "KesslerWatch/0.2 (hackathon prototype)"

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km

OBJECT_TYPES = ("PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN")


# --------------------------------------------------------------------------- env


def _load_env(path: Path) -> None:
    """Minimal KEY=VALUE .env reader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(BASE_DIR / ".env")


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def spacetrack_configured() -> bool:
    return bool(_env("SPACETRACK_USER") and _env("SPACETRACK_PASS"))


def n2yo_configured() -> bool:
    return bool(_env("N2YO_API_KEY"))


# --------------------------------------------------------------------------- rate limiting


class RateLimited(Exception):
    pass


class RateLimiter:
    """Sliding-window limiter over several (max_requests, window_s) rules. Non-blocking."""

    def __init__(self, rules: list[tuple[int, float]]):
        self.rules = rules
        self.hits: deque[float] = deque()
        self.lock = threading.Lock()

    def take(self) -> None:
        with self.lock:
            now = time.monotonic()
            longest = max(w for _, w in self.rules)
            while self.hits and now - self.hits[0] > longest:
                self.hits.popleft()
            for limit, window in self.rules:
                if sum(1 for t in self.hits if now - t <= window) >= limit:
                    raise RateLimited(f"limit of {limit} requests per {int(window)} s reached")
            self.hits.append(now)


# Space-Track documented limits: <30/min and <300/hour. Stay safely under.
ST_LIMITER = RateLimiter([(25, 60), (250, 3600)])
CELESTRAK_LOOKUP_LIMITER = RateLimiter([(20, 60), (200, 3600)])
# N2YO documented limits: 100 transactions/hour each for "above" and "visualpasses".
N2YO_ABOVE_LIMITER = RateLimiter([(90, 3600)])
N2YO_PASSES_LIMITER = RateLimiter([(90, 3600)])
N2YO_POSITIONS_LIMITER = RateLimiter([(900, 3600)])  # documented 1000/hour


# --------------------------------------------------------------------------- TLE helpers


def _epoch_utc(sat: Satrec) -> datetime:
    year = sat.epochyr + (2000 if sat.epochyr < 57 else 1900)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat.epochdays - 1)


def normalize_type(raw: str | None) -> str:
    t = (raw or "").strip().upper()
    return t if t in OBJECT_TYPES else "UNKNOWN"


def infer_type(name: str) -> str:
    """Name-based guess, used only when the source has no official OBJECT_TYPE (CelesTrak TLE groups)."""
    n = name.upper()
    if " DEB" in n or n.endswith("DEB"):
        return "DEBRIS"
    if "R/B" in n:
        return "ROCKET BODY"
    return "PAYLOAD"


def describe(name: str, l1: str, l2: str, *, group: str, object_type: str, type_source: str) -> dict | None:
    """Catalog entry with perigee/apogee (km altitude) from the TLE's mean elements."""
    try:
        sat = Satrec.twoline2rv(l1, l2)
    except Exception:
        return None
    n = sat.no_kozai / 60.0  # rad/s
    if n <= 0:
        return None
    a = (MU_EARTH / n**2) ** (1 / 3)
    e = sat.ecco
    return {
        "norad": int(sat.satnum),
        "name": name.strip() or f"NORAD {sat.satnum}",
        "object_type": object_type,
        "type_source": type_source,  # "space-track" (official) or "inferred" (from name)
        "group": group,
        "tle1": l1,
        "tle2": l2,
        "epoch": _epoch_utc(sat).isoformat().replace("+00:00", "Z"),
        "perigee_km": round(a * (1 - e) - R_EARTH, 1),
        "apogee_km": round(a * (1 + e) - R_EARTH, 1),
    }


def _parse_tle_text(text: str) -> list[tuple[str, str, str]]:
    """Parse 3-line TLE text into (name, line1, line2) tuples."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name.strip().removeprefix("0 "), l1, l2))
            i += 3
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- Space-Track

ST_BASE = "https://www.space-track.org"
ST_LOGIN = f"{ST_BASE}/ajaxauth/login"
_ST_FIELDS = "NORAD_CAT_ID,OBJECT_NAME,OBJECT_TYPE,EPOCH,DECAY_DATE,TLE_LINE1,TLE_LINE2"
# Documented recipe for propagable on-orbit objects: decay_date null + epoch within 10 days.
ST_GP_ALL = (
    f"{ST_BASE}/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-10"
    f"/predicates/{_ST_FIELDS}/orderby/norad_cat_id%20asc/format/json"
)
ST_GP_ONE = f"{ST_BASE}/basicspacedata/query/class/gp/norad_cat_id/{{norad}}/predicates/{_ST_FIELDS}/format/json"

ST_CACHE = CACHE_DIR / "spacetrack_gp.json"
ST_REFRESH_S = 3600  # docs: GP at most once per hour; Sentinel refreshes on that cadence, never faster
# SATCAT, one query per country (Sentinel's default watchlist), cached a day.
ST_SATCAT_COUNTRY = (
    f"{ST_BASE}/basicspacedata/query/class/satcat/COUNTRY/{{country}}/OBJECT_TYPE/PAYLOAD/DECAY/null-val/CURRENT/Y"
    "/predicates/NORAD_CAT_ID,SATNAME,COUNTRY,OBJECT_TYPE,PERIGEE,APOGEE,LAUNCH/orderby/NORAD_CAT_ID%20asc/format/json"
)
SATCAT_TTL_S = 24 * 3600
ST_FAIL_COOLDOWN_S = 15 * 60
ST_SESSION_S = 90 * 60


class SourceError(Exception):
    """Upstream failure with a message that is safe to show to users (no secrets)."""


class SpaceTrack:
    def __init__(self) -> None:
        self.client: httpx.Client | None = None
        self.logged_in_at = 0.0
        self.last_failure = 0.0
        self.last_error: str | None = None
        self.lock = threading.Lock()

    def _client(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=180, headers={"User-Agent": USER_AGENT})
        return self.client

    def _login(self) -> None:
        user, pw = _env("SPACETRACK_USER"), _env("SPACETRACK_PASS")
        if not (user and pw):
            raise SourceError("Space-Track credentials are not configured.")
        ST_LIMITER.take()
        try:
            r = self._client().post(ST_LOGIN, data={"identity": user, "password": pw})
        except httpx.HTTPError:
            raise SourceError("Could not reach Space-Track.") from None
        if r.status_code != 200 or "failed" in r.text.lower():
            raise SourceError("Space-Track login failed (check SPACETRACK_USER / SPACETRACK_PASS).")
        self.logged_in_at = time.time()

    def query(self, url: str) -> list[dict]:
        with self.lock:
            if time.time() - self.logged_in_at > ST_SESSION_S:
                self._login()
            for attempt in (1, 2):
                ST_LIMITER.take()
                try:
                    r = self._client().get(url)
                except httpx.HTTPError:
                    raise SourceError("Could not reach Space-Track.") from None
                if r.status_code == 401 and attempt == 1:
                    self._login()
                    continue
                if r.status_code != 200:
                    raise SourceError(f"Space-Track returned HTTP {r.status_code}.")
                data = r.json()
                if isinstance(data, dict):  # error payloads are objects, results are lists
                    raise SourceError(f"Space-Track error: {str(data)[:200]}")
                return data
            raise SourceError("Space-Track session could not be established.")

    def can_try(self) -> bool:
        return spacetrack_configured() and time.time() - self.last_failure > ST_FAIL_COOLDOWN_S

    def fail(self, msg: str) -> None:
        self.last_failure = time.time()
        self.last_error = msg
        log.warning("Space-Track: %s", msg)


SPACETRACK = SpaceTrack()


def _st_row_to_entry(row: dict, group: str) -> dict | None:
    l1, l2 = row.get("TLE_LINE1"), row.get("TLE_LINE2")
    if not (l1 and l2):
        return None
    return describe(
        row.get("OBJECT_NAME") or "",
        l1,
        l2,
        group=group,
        object_type=normalize_type(row.get("OBJECT_TYPE")),
        type_source="space-track",
    )


def _read_st_cache() -> tuple[float, list[dict]] | None:
    if not ST_CACHE.exists():
        return None
    try:
        blob = json.loads(ST_CACHE.read_text(encoding="utf-8"))
        return float(blob["fetched_at"]), blob["rows"]
    except Exception:
        return None


def spacetrack_catalog() -> tuple[list[dict], float] | None:
    """Full on-orbit GP catalog from Space-Track: fresh cache, else network, else stale cache."""
    cached = _read_st_cache()
    if cached and time.time() - cached[0] < ST_REFRESH_S:
        return cached[1], cached[0]
    if SPACETRACK.can_try():
        try:
            rows = SPACETRACK.query(ST_GP_ALL)
            if not rows:
                raise SourceError("Space-Track returned an empty catalog.")
            now = time.time()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ST_CACHE.write_text(json.dumps({"fetched_at": now, "rows": rows}), encoding="utf-8")
            SPACETRACK.last_error = None
            return rows, now
        except (SourceError, RateLimited) as e:
            SPACETRACK.fail(str(e))
    if cached:
        log.warning("Space-Track: using stale cache from %s", datetime.fromtimestamp(cached[0], timezone.utc))
        return cached[1], cached[0]
    return None


def spacetrack_satcat_payloads(country: str) -> tuple[list[dict], str] | None:
    """On-orbit PAYLOAD rows of one SATCAT COUNTRY code (e.g. IND). Cached 24 h on disk; stale cache on failure.

    Returns (rows, fetched_utc) or None when neither Space-Track nor a cache is available.
    """
    path = CACHE_DIR / f"spacetrack_satcat_{country.lower()}.json"
    cached = None
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            cached = (float(blob["fetched_at"]), blob["rows"])
        except Exception:
            cached = None
    if cached and time.time() - cached[0] < SATCAT_TTL_S:
        return cached[1], _utc(cached[0])
    if SPACETRACK.can_try():
        try:
            rows = SPACETRACK.query(ST_SATCAT_COUNTRY.format(country=country))
            now = time.time()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fetched_at": now, "rows": rows}), encoding="utf-8")
            return rows, _utc(now)
        except (SourceError, RateLimited) as e:
            SPACETRACK.fail(str(e))
    return (cached[1], _utc(cached[0])) if cached else None


def celestrak_active_norads() -> set[int] | None:
    """NORAD IDs in CelesTrak's "active" group (cached 2 h like every CelesTrak group), or None if unavailable."""
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        text = _celestrak_group("active", client)
    if not text:
        return None
    out = set()
    for _, l1, l2 in _parse_tle_text(text):
        try:
            out.add(int(Satrec.twoline2rv(l1, l2).satnum))
        except Exception:
            continue
    return out


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- CelesTrak fallback

CELESTRAK_GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CELESTRAK_CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=tle"
CELESTRAK_TTL_S = 2 * 3600
CELESTRAK_GROUPS = [
    "stations",
    "active",
    "cosmos-1408-debris",
    "fengyun-1c-debris",
    "iridium-33-debris",
    "cosmos-2251-debris",
]


def _celestrak_group(group: str, client: httpx.Client) -> str | None:
    """Fresh cache, else network, else stale cache. Never fetched per request."""
    path = CACHE_DIR / f"{group}.txt"
    if path.exists() and time.time() - path.stat().st_mtime < CELESTRAK_TTL_S:
        return path.read_text()
    try:
        r = client.get(CELESTRAK_GROUP_URL.format(group=group))
        r.raise_for_status()
        if not _parse_tle_text(r.text):
            raise ValueError("no TLEs in response")
        path.write_text(r.text)
        return r.text
    except Exception as e:
        log.warning("CelesTrak %s fetch failed: %s", group, type(e).__name__)
        return path.read_text() if path.exists() else None


def _read_sample() -> dict[str, list[tuple[str, str, str]]]:
    groups: dict[str, list[tuple[str, str, str]]] = {}
    if not SAMPLE_FILE.exists():
        return groups
    current = "unknown"
    buf: list[str] = []

    def flush() -> None:
        if buf:
            groups.setdefault(current, []).extend(_parse_tle_text("\n".join(buf)))
            buf.clear()

    for ln in SAMPLE_FILE.read_text().splitlines():
        if ln.startswith("# GROUP "):
            flush()
            current = ln[len("# GROUP ") :].strip()
        else:
            buf.append(ln)
    flush()
    return groups


def _write_sample(groups: dict[str, list[tuple[str, str, str]]]) -> None:
    parts = []
    for group, tles in groups.items():
        parts.append(f"# GROUP {group}")
        for name, l1, l2 in tles:
            parts.extend([name, l1, l2])
    SAMPLE_FILE.write_text("\n".join(parts) + "\n")


def celestrak_catalog() -> tuple[list[dict], bool]:
    """CelesTrak groups (types inferred from names). Returns (entries, used_offline_sample)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[tuple[str, str, str]]] = {}
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for g in CELESTRAK_GROUPS:
            text = _celestrak_group(g, client)
            if text:
                result[g] = _parse_tle_text(text)
    used_sample = False
    if len(result) == len(CELESTRAK_GROUPS):
        if not SAMPLE_FILE.exists():
            _write_sample(result)
    else:
        sample = _read_sample()
        for g in CELESTRAK_GROUPS:
            if g not in result and g in sample:
                log.warning("CelesTrak: using offline sample for %s", g)
                result[g] = sample[g]
                used_sample = True
    entries = []
    for group, tles in result.items():
        for name, l1, l2 in tles:
            e = describe(name, l1, l2, group=group, object_type=infer_type(name), type_source="inferred")
            if e:
                entries.append(e)
    return entries, used_sample


# --------------------------------------------------------------------------- catalog


def _dedupe(entries: list[dict]) -> list[dict]:
    seen: set[int] = set()
    out = []
    for e in entries:
        if e and e["norad"] not in seen:
            seen.add(e["norad"])
            out.append(e)
    return out


def build_catalog() -> tuple[list[dict], dict]:
    """Space-Track full catalog if possible, otherwise CelesTrak groups. Returns (entries, meta)."""
    if spacetrack_configured():
        st = spacetrack_catalog()
        if st:
            rows, fetched_at = st
            entries = _dedupe([_st_row_to_entry(r, "space-track") for r in rows])
            return entries, {
                "source": "space-track",
                "fetched_utc": datetime.fromtimestamp(fetched_at, timezone.utc).isoformat().replace("+00:00", "Z"),
                "type_source": "official",
                "note": SPACETRACK.last_error,
            }
    entries, used_sample = celestrak_catalog()
    reason = (
        "Space-Track credentials missing" if not spacetrack_configured() else (SPACETRACK.last_error or "Space-Track unavailable")
    )
    return _dedupe(entries), {
        "source": "celestrak-sample" if used_sample else "celestrak",
        "fetched_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "type_source": "inferred",
        "note": f"Fallback: {reason}. Object types inferred from names.",
    }


# --------------------------------------------------------------------------- single-object lookup

LOOKUP_TTL_S = 4 * 3600
MISS_TTL_S = 10 * 60
_lookup_cache: dict[int, tuple[float, dict | None, str | None]] = {}


class NotFound(Exception):
    pass


class Decayed(Exception):
    def __init__(self, decay_date: str):
        super().__init__(decay_date)
        self.decay_date = decay_date


def lookup_object(norad: int) -> dict:
    """Fetch one object by NORAD ID: Space-Track first, CelesTrak CATNR fallback. Cached."""
    hit = _lookup_cache.get(norad)
    if hit and time.time() - hit[0] < (LOOKUP_TTL_S if hit[1] else MISS_TTL_S):
        if hit[1]:
            return hit[1]
        if hit[2]:
            raise Decayed(hit[2])
        raise NotFound()

    errors: list[str] = []
    if SPACETRACK.can_try():
        try:
            rows = SPACETRACK.query(ST_GP_ONE.format(norad=norad))
            if rows:
                row = rows[0]
                if row.get("DECAY_DATE"):
                    _lookup_cache[norad] = (time.time(), None, row["DECAY_DATE"])
                    raise Decayed(row["DECAY_DATE"])
                entry = _st_row_to_entry(row, "lookup")
                if entry:
                    _lookup_cache[norad] = (time.time(), entry, None)
                    return entry
        except (SourceError, RateLimited) as e:
            errors.append(str(e))

    try:
        CELESTRAK_LOOKUP_LIMITER.take()
        with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(CELESTRAK_CATNR_URL.format(norad=norad))
        tles = _parse_tle_text(r.text) if r.status_code == 200 else []
        if tles:
            name, l1, l2 = tles[0]
            entry = describe(name, l1, l2, group="lookup", object_type=infer_type(name), type_source="inferred")
            if entry:
                _lookup_cache[norad] = (time.time(), entry, None)
                return entry
    except RateLimited as e:
        errors.append(f"CelesTrak lookup {e}")
    except httpx.HTTPError:
        errors.append("Could not reach CelesTrak.")

    if errors:
        raise SourceError("; ".join(errors))
    _lookup_cache[norad] = (time.time(), None, None)
    raise NotFound()


# --------------------------------------------------------------------------- N2YO

N2YO_BASE = "https://api.n2yo.com/rest/v1/satellite"
N2YO_ABOVE_TTL_S = 60
N2YO_PASSES_TTL_S = 30 * 60
_n2yo_cache: dict[tuple, tuple[float, dict]] = {}
_n2yo_client = httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT})


class N2YOError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _n2yo_get(path: str, limiter: RateLimiter) -> dict:
    """GET {N2YO_BASE}/{path}&apiKey=... (documented form). Errors never include the URL or key."""
    key = _env("N2YO_API_KEY")
    if not key:
        raise N2YOError(503, "N2YO API key is not configured on the server (N2YO_API_KEY in backend/.env).")
    try:
        limiter.take()
    except RateLimited:
        raise N2YOError(429, "N2YO hourly request limit reached. Try again later.") from None
    try:
        r = _n2yo_client.get(f"{N2YO_BASE}/{path}&apiKey={key}")
    except httpx.HTTPError:
        raise N2YOError(502, "Could not reach N2YO.") from None
    if r.status_code != 200:
        raise N2YOError(502, f"N2YO returned HTTP {r.status_code}.")
    try:
        data = r.json()
    except ValueError:
        raise N2YOError(502, "N2YO returned an unreadable response.") from None
    if isinstance(data, dict) and data.get("error"):
        raise N2YOError(502, f"N2YO error: {str(data['error']).replace(key, '***')}")
    return data


def _cached(key: tuple, ttl: float, fetch) -> tuple[dict, bool]:
    hit = _n2yo_cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1], True
    data = fetch()
    _n2yo_cache[key] = (time.time(), data)
    return data, False


def n2yo_above(lat: float, lon: float, alt_m: float, radius_deg: int) -> dict:
    key = ("above", round(lat, 2), round(lon, 2), round(alt_m), radius_deg)
    data, cached = _cached(
        key,
        N2YO_ABOVE_TTL_S,
        lambda: _n2yo_get(f"above/{lat:.4f}/{lon:.4f}/{alt_m:.0f}/{radius_deg}/0/", N2YO_ABOVE_LIMITER),
    )
    sats = [
        {
            "norad": int(s["satid"]),
            "name": s.get("satname", ""),
            "int_designator": s.get("intDesignator"),
            "launch_date": s.get("launchDate"),
            "lat": s.get("satlat"),
            "lon": s.get("satlng"),
            "alt_km": s.get("satalt"),
        }
        for s in data.get("above") or []
    ]
    return {"count": len(sats), "satellites": sats, "cached": cached}


def n2yo_passes(norad: int, lat: float, lon: float, alt_m: float, days: int, min_visibility_s: int) -> dict:
    key = ("passes", norad, round(lat, 2), round(lon, 2), round(alt_m), days, min_visibility_s)
    data, cached = _cached(
        key,
        N2YO_PASSES_TTL_S,
        lambda: _n2yo_get(
            f"visualpasses/{norad}/{lat:.4f}/{lon:.4f}/{alt_m:.0f}/{days}/{min_visibility_s}/", N2YO_PASSES_LIMITER
        ),
    )

    def iso(ts) -> str | None:
        return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z") if ts else None

    passes = [
        {
            "start_utc": iso(p.get("startUTC")),
            "start_az": p.get("startAz"),
            "start_az_compass": p.get("startAzCompass"),
            "max_utc": iso(p.get("maxUTC")),
            "max_el": p.get("maxEl"),
            "max_az_compass": p.get("maxAzCompass"),
            "end_utc": iso(p.get("endUTC")),
            "end_az_compass": p.get("endAzCompass"),
            "mag": p.get("mag"),
            "duration_s": p.get("duration"),
        }
        for p in data.get("passes") or []
    ]
    info = data.get("info") or {}
    return {"norad": norad, "name": info.get("satname"), "count": len(passes), "passes": passes, "cached": cached}


def n2yo_position(norad: int, lat: float = 0.0, lon: float = 0.0) -> dict:
    """Current sub-satellite point from N2YO (used for verification only)."""
    data = _n2yo_get(f"positions/{norad}/{lat:.4f}/{lon:.4f}/0/1/", N2YO_POSITIONS_LIMITER)
    p = (data.get("positions") or [{}])[0]
    return {"lat": p.get("satlatitude"), "lon": p.get("satlongitude"), "alt_km": p.get("sataltitude"), "timestamp": p.get("timestamp")}


# --------------------------------------------------------------------------- space weather (NOAA SWPC)

SWPC_F107 = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
SWPC_27DAY = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"
OUTLOOK_CACHE = CACHE_DIR / "swpc_27day.json"
OUTLOOK_TTL_S = 6 * 3600  # issued weekly
_outlook_mem: tuple[float, dict] | None = None
SWPC_KP = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SW_CACHE = CACHE_DIR / "spaceweather.json"
SW_TTL_S = 30 * 60
_sw_mem: tuple[float, dict] | None = None


def _kp_rows(data) -> list[tuple[str, float, float | None]]:
    """NOAA's Kp product has been served both as a header+rows array and as a list of objects."""
    rows = []
    if data and isinstance(data[0], list):  # [["time_tag","Kp","a_running",...], [...], ...]
        head = [str(h) for h in data[0]]
        ti, ki = head.index("time_tag"), head.index("Kp")
        ai = head.index("a_running") if "a_running" in head else None
        for r in data[1:]:
            rows.append((r[ti], float(r[ki]), float(r[ai]) if ai is not None and r[ai] is not None else None))
    else:
        for r in data:
            if r.get("Kp") is not None:
                a = r.get("a_running")
                rows.append((r["time_tag"], float(r["Kp"]), float(a) if a is not None else None))
    return sorted(rows)


def _fetch_space_weather() -> dict:
    with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
        f = c.get(SWPC_F107)
        k = c.get(SWPC_KP)
    f.raise_for_status()
    k.raise_for_status()
    flux = sorted((r for r in f.json() if r.get("flux") is not None), key=lambda r: r["time_tag"])
    kp = _kp_rows(k.json())
    if not flux or not kp:
        raise ValueError("empty SWPC response")
    a_last = [a for _, _, a in kp[-8:] if a is not None]
    # Observed daily flux over the last 27 days (several readings per day: one value per day, the last of the day).
    by_day: dict[str, float] = {}
    for r in flux:
        by_day[r["time_tag"][:10]] = float(r["flux"])
    recent = [by_day[d] for d in sorted(by_day)[-27:]]
    return {
        "f107_27d_mean": round(sum(recent) / len(recent), 1) if recent else None,
        "f107_27d_days": len(recent),
        "f107": float(flux[-1]["flux"]),
        "f107_time_utc": flux[-1]["time_tag"] + "Z",
        "kp": kp[-1][1],
        "kp_time_utc": kp[-1][0] + "Z",
        # Daily-style Ap: mean of the last eight 3-hourly ap values (24 h).
        "ap": round(sum(a_last) / len(a_last), 1) if a_last else None,
        "ap_basis": "mean of last 8 three-hourly ap values (24 h)",
        "source": "NOAA SWPC",
    }


def _parse_27day(text: str) -> dict:
    """NOAA SWPC 27-day outlook table: one row per day `YYYY Mon DD  flux  Ap  Kp`."""
    issued = None
    days = []
    for ln in text.splitlines():
        if ln.startswith(":Issued:"):
            issued = ln.split(":Issued:", 1)[1].strip()
        parts = ln.split()
        if len(parts) >= 6 and parts[0].isdigit() and len(parts[0]) == 4:
            try:
                d = datetime.strptime(" ".join(parts[:3]), "%Y %b %d").date()
                days.append({"date": d.isoformat(), "f107": float(parts[3]), "ap": float(parts[4]), "kp_max": float(parts[5])})
            except ValueError:
                continue
    if not days:
        raise ValueError("no rows in the 27-day outlook")
    return {"issued": issued, "days": days, "source": "NOAA SWPC 27-day outlook"}


def solar_outlook() -> dict:
    """NOAA SWPC 27-day forecast of F10.7 / Ap (issued weekly). Cached 6 h; stale cache on failure."""
    global _outlook_mem
    if _outlook_mem and time.time() - _outlook_mem[0] < OUTLOOK_TTL_S:
        return {**_outlook_mem[1], "stale": False}
    try:
        with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(SWPC_27DAY)
        r.raise_for_status()
        data = _parse_27day(r.text)
        now = time.time()
        _outlook_mem = (now, data)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTLOOK_CACHE.write_text(json.dumps({"fetched_at": now, "data": data}), encoding="utf-8")
        return {**data, "stale": False}
    except Exception as e:
        log.warning("SWPC 27-day outlook fetch failed: %s", type(e).__name__)
        if _outlook_mem:
            return {**_outlook_mem[1], "stale": True}
        if OUTLOOK_CACHE.exists():
            blob = json.loads(OUTLOOK_CACHE.read_text(encoding="utf-8"))
            _outlook_mem = (float(blob["fetched_at"]), blob["data"])
            return {**blob["data"], "stale": True}
        raise SourceError("NOAA SWPC 27-day outlook is unavailable.") from None


def space_weather() -> dict:
    """Latest F10.7 / Kp / Ap, cached 30 min in memory and on disk; stale cache on failure."""
    global _sw_mem
    if _sw_mem and time.time() - _sw_mem[0] < SW_TTL_S:
        return {**_sw_mem[1], "stale": False}
    try:
        data = _fetch_space_weather()
        now = time.time()
        data["updated"] = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        _sw_mem = (now, data)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SW_CACHE.write_text(json.dumps({"fetched_at": now, "data": data}), encoding="utf-8")
        return {**data, "stale": False}
    except Exception as e:
        log.warning("Space weather fetch failed: %s", type(e).__name__)
        if _sw_mem:
            return {**_sw_mem[1], "stale": True}
        if SW_CACHE.exists():
            blob = json.loads(SW_CACHE.read_text(encoding="utf-8"))
            _sw_mem = (float(blob["fetched_at"]), blob["data"])
            return {**blob["data"], "stale": True}
        raise SourceError("NOAA SWPC space weather is unavailable.") from None
