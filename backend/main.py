"""KesslerWatch API."""
from __future__ import annotations

import logging
import statistics
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from pydantic import BaseModel, Field

import ml
import orbit
import risk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="KesslerWatch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# The full catalog is several MB of JSON; gzip cuts it to roughly a third.
app.add_middleware(GZipMiddleware, minimum_size=2048)

# Default observer: Hubballi (Hubli), Karnataka. Elevation ~627 m (GeoNames via Open-Meteo).
DEFAULT_NAME = "Hubballi"
DEFAULT_LAT = 15.3647
DEFAULT_LON = 75.1240
DEFAULT_ALT_M = 627.0
CATALOG_TTL_S = 2 * 3600  # rebuild from the (separately cached) sources at most this often

_catalog: list[dict] = []
_by_norad: dict[int, dict] = {}
_meta: dict = {}
_built_at = 0.0
_lock = threading.Lock()
_index: risk.CatalogIndex | None = None
_index_built_at = -1.0
_index_lock = threading.Lock()
_last_screen: dict | None = None


def get_catalog() -> tuple[list[dict], dict]:
    global _catalog, _by_norad, _meta, _built_at
    with _lock:
        if not _catalog or time.time() - _built_at > CATALOG_TTL_S:
            t0 = time.time()
            _catalog, _meta = orbit.build_catalog()
            _by_norad = {e["norad"]: e for e in _catalog}
            _built_at = time.time()
            logging.getLogger("kesslerwatch").info(
                "catalog: %d objects from %s in %.1f s", len(_catalog), _meta.get("source"), time.time() - t0
            )
        return _catalog, _meta


def get_index() -> risk.CatalogIndex:
    """Parsed satrecs for screening, rebuilt only when the catalog is rebuilt."""
    global _index, _index_built_at
    get_catalog()
    with _index_lock:
        if _index is None or _index_built_at != _built_at:
            _index = risk.build_index(_catalog)
            _index_built_at = _built_at
        return _index


def resolve_object(norad: int) -> dict:
    """Catalog entry for any NORAD ID (catalog first, then on-demand lookup)."""
    get_catalog()
    if norad in _by_norad:
        return _by_norad[norad]
    try:
        return orbit.lookup_object(norad)
    except orbit.Decayed as d:
        raise HTTPException(410, f"NORAD {norad} re-entered (decay date {d.decay_date}); no current orbit.") from None
    except orbit.NotFound:
        raise HTTPException(404, f"No element set found for NORAD {norad}.") from None
    except orbit.SourceError as e:
        raise HTTPException(502, f"Lookup failed: {e}") from None


@app.on_event("startup")
def _warm() -> None:
    # Build in the background so the first page load doesn't wait on a cold download.
    threading.Thread(target=get_catalog, daemon=True).start()


@app.get("/catalog")
def catalog() -> list[dict]:
    return get_catalog()[0]


@app.get("/status")
def status() -> dict:
    """Catalog provenance and which optional integrations are configured (never the secrets)."""
    entries, meta = get_catalog()
    counts: dict[str, int] = {t: 0 for t in orbit.OBJECT_TYPES}
    for e in entries:
        counts[e["object_type"]] += 1
    return {
        "objects": len(entries),
        "by_type": counts,
        "catalog": meta,
        "spacetrack_configured": orbit.spacetrack_configured(),
        "n2yo_configured": orbit.n2yo_configured(),
        "default_observer": {"lat": DEFAULT_LAT, "lon": DEFAULT_LON, "alt_m": DEFAULT_ALT_M, "name": DEFAULT_NAME},
    }


@app.get("/object/{norad}")
def object_by_norad(norad: int) -> dict:
    if not 0 < norad < 1_000_000:
        raise HTTPException(422, "NORAD ID must be a positive integer.")
    return resolve_object(norad)


class ScreenRequest(BaseModel):
    norad: int = Field(gt=0)
    horizon_h: float = Field(24, gt=0, le=72)
    screen_km: float = Field(10, gt=0, le=50)
    hbr_m: float = Field(20, gt=0, le=500)


@app.post("/screen")
def screen(req: ScreenRequest) -> list[dict]:
    """Conjunctions of one asset against the full catalog (see risk.screen and CLAUDE.md physics rules)."""
    global _last_screen
    primary = resolve_object(req.norad)
    try:
        results, stats = risk.screen(get_index(), primary, req.horizon_h, req.screen_km, req.hbr_m)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    # AI risk: the live model's estimate of the FINAL risk, from ESA CDM data (a model estimate, see ml.py).
    try:
        ml.score_conjunctions(results, datetime.fromisoformat(stats["window_start_utc"].replace("Z", "+00:00")))
    except Exception:  # never let the optional model break screening
        logging.getLogger("kesslerwatch").exception("AI risk scoring failed")
    _last_screen = {**stats, "primary_name": primary["name"]}
    logging.getLogger("kesslerwatch").info(
        "screen %s: %d conjunctions, %d alerts, %.1f s, %s pairs/s",
        primary["name"], stats["conjunctions"], stats["alerts"], stats["seconds"], stats["pairs_per_sec"],
    )
    return results


class ManeuverRequest(BaseModel):
    norad: int = Field(gt=0)
    secondary: int = Field(gt=0)
    tca_utc: datetime
    hbr_m: float = Field(20, gt=0, le=500)


@app.post("/maneuver")
def maneuver(req: ManeuverRequest) -> list[dict]:
    """Along-track burn options (Δs ≈ 3·Δv·Δt) that bring Pc below the target. Estimates."""
    tca = req.tca_utc if req.tca_utc.tzinfo else req.tca_utc.replace(tzinfo=timezone.utc)
    try:
        return risk.maneuver(resolve_object(req.norad), resolve_object(req.secondary), tca.astimezone(timezone.utc), req.hbr_m)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/reentry")
def reentry(limit: int = Query(25, ge=1, le=200)) -> list[dict]:
    """Objects likely to re-enter soon. HEURISTIC from the mean-motion derivative; every value is est."""
    return risk.reentry(get_catalog()[0], limit=limit)


@app.get("/spaceweather")
def spaceweather() -> dict:
    try:
        return orbit.space_weather()
    except orbit.SourceError as e:
        raise HTTPException(502, str(e)) from None


@app.get("/model/metrics")
def model_metrics() -> dict:
    """Training/evaluation report of the AI risk model (baseline vs full vs live), written by `python ml.py train`."""
    m = ml.model_metrics()
    if m is None:
        raise HTTPException(404, "The AI risk model has not been trained yet. Run `python ml.py train` in backend/.")
    return m


@app.get("/metrics")
def metrics() -> dict:
    entries, _ = get_catalog()
    now = datetime.now(timezone.utc)
    ages = [(now - datetime.fromisoformat(e["epoch"].replace("Z", "+00:00"))).total_seconds() / 3600 for e in entries]
    return {
        "objects_tracked": len(entries),
        "pairs_per_sec": _last_screen["pairs_per_sec"] if _last_screen else None,
        "median_tle_age_h": round(statistics.median(ages), 1) if ages else None,
        # Full model, calibrated high-risk probability, on the held-out events (see /model/metrics).
        "brier_score": (ml.model_metrics() or {}).get("results", {}).get("holdout", {}).get("full", {}).get("brier"),
        "last_screen": _last_screen,
    }


def _n2yo_call(fn, *args):
    try:
        return fn(*args)
    except orbit.N2YOError as e:
        raise HTTPException(e.status, e.message) from None


@app.get("/above")
def above(
    lat: float = Query(DEFAULT_LAT, ge=-90, le=90),
    lon: float = Query(DEFAULT_LON, ge=-180, le=180),
    radius: int = Query(70, ge=0, le=90, description="Search radius in degrees above the horizon (N2YO: 0-90)"),
    alt_m: float = Query(DEFAULT_ALT_M, ge=-500, le=9000),
) -> dict:
    """Satellites currently above an observer (N2YO 'what's up', all categories)."""
    data = _n2yo_call(orbit.n2yo_above, lat, lon, alt_m, radius)
    return {"observer": {"lat": lat, "lon": lon, "alt_m": alt_m}, "radius_deg": radius, **data}


@app.get("/passes/{norad}")
def passes(
    norad: int,
    lat: float = Query(DEFAULT_LAT, ge=-90, le=90),
    lon: float = Query(DEFAULT_LON, ge=-180, le=180),
    alt_m: float = Query(DEFAULT_ALT_M, ge=-500, le=9000),
    days: int = Query(10, ge=1, le=10),
    min_visibility: int = Query(60, ge=1, le=600, description="Minimum optically visible seconds"),
) -> dict:
    """Upcoming optically visible passes over an observer (N2YO 'visual passes')."""
    data = _n2yo_call(orbit.n2yo_passes, norad, lat, lon, alt_m, days, min_visibility)
    return {"observer": {"lat": lat, "lon": lon, "alt_m": alt_m}, **data}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
